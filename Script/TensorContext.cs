using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;

namespace ShaderGPT {
public readonly struct TexView : System.IEquatable<TexView> {
	public readonly (Texture t, int s0, int s1, int o0, int o1) data;
	public TexView(Texture tex, int size0, int size1, int off0=0, int off1=0) {
		data = (tex, size0, size1, off0, off1);
	}
	public static implicit operator TexView(Texture tex) => tex ? new TexView(tex,
		(tex.height >> (tex.mipmapCount-1)) >> (int)tex.mipMapBias,
		(tex.width  >> (tex.mipmapCount-1)) << (int)tex.mipMapBias) : default;
	public static explicit operator Texture(TexView view) => view.data.t;
	public bool Equals(TexView other) => data == other.data;
	public static implicit operator bool(TexView view) => view.data != default;
}
public class TensorContext {
	// tensor properties and i/o
	public int Offset0(TexView view) => view.data.o0;
	public int Offset1(TexView view) => view.data.o1;
	public int Size0(TexView view) => view.data.s0;
	public int Size1(TexView view) => view.data.s1;
	public int Tile1(TexView view) => (int)view.data.t.mipMapBias;
	public int Lod(TexView view) => (view.data.t.mipmapCount-1);
	public Vector2Int Size(TexView view) => new Vector2Int(Size0(view), Size1(view));
	public VertexAttributeFormat DType(TexView view) => graphicsFormatToDType[view.data.t.graphicsFormat];
	public TexView Slice(Texture tex, int size0, int size1, int off0=0, int off1=0) =>
		new TexView(tex, size0, size1, off0, off1);

	public void FixSize0(Texture tex, int size0) {
		var h = tex.height >> (tex.mipmapCount-1);
		Debug.Assert(h % size0 == 0);
		tex.mipMapBias = Mathf.Log((float)h / size0, 2);
	}
	public NativeArray<float> GetData(Texture2D tex) {
		return tex.GetRawTextureData<float>();
	}
	public void SetData(Texture2D tex, float[] data) {
		tex.SetPixelData(data, 0, 0);
		tex.Apply(updateMipmaps:false, makeNoLongerReadable:false);
	}
	static Texture Copy(Texture2D output, RenderTexture input, int lod) {
		var active = RenderTexture.active;
		if(lod != 0) {
			// copy last mip level for ReadPixels
			var desc = input.descriptor;
			desc.width >>= lod;
			desc.height >>= lod;
			desc.useMipMap = false;
			var clone = RenderTexture.GetTemporary(desc);
			RenderTexture.active = clone; // must init before copy, but clone.Create() doesn't work somehow
			Graphics.CopyTexture(input, 0, lod, clone, 0, 0);
			input = clone;
		}
		RenderTexture.active = input;
		output.ReadPixels(new Rect(0, 0, output.width, output.height), 0, 0);
		if(lod != 0)
			RenderTexture.ReleaseTemporary(input);
		RenderTexture.active = active;
		return output;
	}

	// tensor creation
	HashSet<Texture> texSet = new HashSet<Texture>();
	Dictionary<string,RenderTexture> rtDict = new Dictionary<string,RenderTexture>();
	public virtual RenderTexture PersistentGPUTensor(string name, int size0, int size1, VertexAttributeFormat? dtype=null, int lod=0) {
		if(rtDict.ContainsKey(name)) {
			var rt = rtDict[name];
			Debug.Assert(size0 == Size0(rt) && size1 == Size1(rt) && (dtype??defaultDType) == DType(rt) && lod == Lod(rt));
			return rt;
		}
		var tex = RenderTexture.GetTemporary(GPUTensorDescriptor(size0, size1, dtype:dtype, lod:lod));
		Debug.Assert(!texSet.Contains(tex));
		rtDict[name] = tex;
		FixSize0(tex, size0);
		return tex;
	}
	public virtual RenderTexture GPUTensor(int size0, int size1, VertexAttributeFormat? dtype=null, int lod=0, bool autoGenMips=true) {
		// NOTE: autoGenMips is ignored because deferred generation is not implemented
		var tex = RenderTexture.GetTemporary(GPUTensorDescriptor(size0, size1, dtype:dtype, lod:lod));
		Debug.Assert(!texSet.Contains(tex));
		texSet.Add(tex);
		FixSize0(tex, size0);
		return tex;
	}
	public virtual Texture2D CPUTensor(int size0, int size1, VertexAttributeFormat? dtype=null) {
		var (width, height, textureFormat) = CPUTensorDescriptor(size0, size1, dtype:dtype);
		var tex = new Texture2D(width, height, textureFormat, mipChain:false, linear:true);
		Debug.Assert(!texSet.Contains(tex));
		texSet.Add(tex);
		FixSize0(tex, size0);
		return tex;
	}
	public virtual void Release(Texture tex) {
		Debug.Assert(tex && texSet.Contains(tex));
		texSet.Remove(tex);
		var rt = tex as RenderTexture;
		if(rt) {
			RenderTexture.ReleaseTemporary(rt);
			return;
		}
		Object.Destroy(tex);
	}
	public virtual void ReleasePersistent() {
		foreach(var pair in rtDict)
			RenderTexture.ReleaseTemporary(pair.Value);
		rtDict.Clear();
	}

	// operator creation
	public virtual Material Operator(Shader shader) {
		return new Material(shader);
	}
	public virtual void Blit(RenderTexture rt, Material mat) {
		Graphics.Blit(null, rt, mat, 0);
		Object.Destroy(mat);
	}

	// utilities
	public int TensorCount() {
		return texSet.Count;
	}
	public float[] GetData(RenderTexture input) {
		var clone = CPUTensor(input.height>>Lod(input), input.width>>Lod(input), dtype:VertexAttributeFormat.Float32);
		Copy(clone, input, Lod(input));
		var data = GetData(clone);
		var output = new float[data.Length];
		var dim0 = clone.height>>Tile1(input);
		var dim1 = 1<<Tile1(input);
		var dim2 = clone.width;
		if(dim1 == 1)
			data.CopyTo(output);
		else { // transpose dim1 & dim2
			var p = 0;
			for(int i=0; i<dim0; i++)
			for(int k=0; k<dim2; k++)
			for(int j=0; j<dim1; j++) {
				var q = ((i*dim1+j)*dim2+k)*4;
				output[p++] = data[q++];
				output[p++] = data[q++];
				output[p++] = data[q++];
				output[p++] = data[q++];
			}
		}
		Release(clone);
		return output;
	}
	public void DebugTensor(Texture input, bool full=false) {
		if(input is RenderTexture rt) {
			var data = new NativeArray<float>(GetData(rt), Allocator.Temp);
			DebugTensor(data, Size0(rt), Size1(rt)*4, full);
			data.Dispose();
		} else if(input is Texture2D tex) {
			var data = GetData(tex);
			DebugTensor(data, Size0(tex), Size1(tex)*4, full);
		} else
			Debug.LogError($"unsupported texture type {input}");
	}
	void DebugTensor(NativeArray<float> data, int nrow, int ncol, bool full) {
		var sb = new System.Text.StringBuilder();
		sb.AppendFormat("tensor(shape=({0}, {1}), [\n", nrow, ncol);
		for(int i=0; i<nrow; i++) {
			if(nrow > 12 && !full && 3 <= i && i < nrow-3) {
				if(i == 3)
					sb.AppendLine("...,");
				continue;
			}
			var row = new NativeSlice<float>(data, i*ncol, ncol);
			sb.Append("[");
			if(ncol > 12 && !full)
				sb.AppendFormat("{0:F4}, {1:F4}, {2:F4},  ..., {3:F4}, {4:F4}, {5:F4}",
					row[0], row[1], row[2], row[ncol-3], row[ncol-2], row[ncol-1]);
			else
				sb.AppendJoin(", ", row.Select(x => x.ToString("F4")));
			sb.AppendLine("],");
		}
		Debug.Log(sb.ToString());
	}

	const int maxTextureSize = 16384;
	public bool autoTile = true;
	public VertexAttributeFormat defaultDType = VertexAttributeFormat.Float32;
	protected (int,int,TextureFormat) CPUTensorDescriptor(int size0, int size1, int size2=4, VertexAttributeFormat? dtype=null) {
		return (size1, size0, dtypeToTexFormat[(dtype??defaultDType,size2)]);
	}
	protected RenderTextureDescriptor GPUTensorDescriptor(int size0, int size1, int size2=4, VertexAttributeFormat? dtype=null, int lod=0, bool autoGenMips=true) {
		if(autoTile && (size0<<lod) == 1 && size1 % 2 == 0) {
			// tile when height==1, which inactivates half wave lanes
			size0 <<= 1;
			size1 >>= 1;
		}
		if(size1 > maxTextureSize) {
			// tile when width exceeds unity limit
			var lvl = Mathf.CeilToInt(Mathf.Log((float)size1/maxTextureSize, 2));
			size1 >>= lvl;
			size0 <<= lvl;
		}
		var desc = new RenderTextureDescriptor(size1, size0, dtypeToRTFormat[(dtype??defaultDType,size2)], 0);
		while(lod > 0 && ((desc.width << lod) > maxTextureSize || (desc.height << lod) > maxTextureSize))
			lod --;
		if(lod > 0) {
			desc.width <<= lod;
			desc.height <<= lod;
			desc.useMipMap = true;
			desc.autoGenerateMips = autoGenMips;
			desc.mipCount = 1+lod;
		}
		return desc;
	}
	static Dictionary<(VertexAttributeFormat, int), TextureFormat> dtypeToTexFormat =
		new Dictionary<(VertexAttributeFormat, int), TextureFormat> {
			{(VertexAttributeFormat.Float32, 4), TextureFormat.RGBAFloat},
			{(VertexAttributeFormat.Float16, 4), TextureFormat.RGBAHalf},
		};
	static Dictionary<(VertexAttributeFormat, int), RenderTextureFormat> dtypeToRTFormat =
		new Dictionary<(VertexAttributeFormat, int), RenderTextureFormat>{
			{(VertexAttributeFormat.Float32, 4), RenderTextureFormat.ARGBFloat},
			{(VertexAttributeFormat.Float16, 4), RenderTextureFormat.ARGBHalf},
		};
	static Dictionary<GraphicsFormat, VertexAttributeFormat> graphicsFormatToDType =
		new Dictionary<GraphicsFormat, VertexAttributeFormat>{
			{GraphicsFormat.R32G32B32A32_SFloat, VertexAttributeFormat.Float32},
			{GraphicsFormat.R16G16B16A16_SFloat, VertexAttributeFormat.Float16},
		};
}
}