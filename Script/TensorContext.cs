using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;

namespace ShaderGPT {
public class TensorContext {
	// tensor properties and i/o
	public int Size0(Texture tex) => tex.height >> (tex.mipmapCount-1);
	public int Size1(Texture tex) => tex.width >> (tex.mipmapCount-1);
	public int Mipmap(Texture tex) => (tex.mipmapCount-1);
	public Vector2Int Size(Texture tex) => new Vector2Int(Size0(tex), Size1(tex));
	public VertexAttributeFormat DType(Texture tex) => graphicsFormatToDType[tex.graphicsFormat];
	public NativeArray<float> GetData(Texture2D tex) {
		return tex.GetRawTextureData<float>();
	}
	public void SetData(Texture2D tex, float[] data) {
		tex.SetPixelData(data, 0, 0);
		tex.Apply(updateMipmaps:false, makeNoLongerReadable:false);
	}
	public Texture Copy(Texture2D output, RenderTexture input, Vector2Int size, Vector2Int outputOffset=default, Vector2Int inputOffset=default) {
		var mipmap = Mipmap(input);
		if(mipmap != 0) {
			// copy last mip level for ReadPixels
			var desc = input.descriptor;
			desc.width >>= mipmap;
			desc.height >>= mipmap;
			desc.useMipMap = false;
			var clone = RenderTexture.GetTemporary(desc);
			Graphics.CopyTexture(input, 0, mipmap, clone, 0, 0);
			Copy(output, clone, size, outputOffset:outputOffset, inputOffset:inputOffset);
			RenderTexture.ReleaseTemporary(clone);
			return output;
		}
		var active = RenderTexture.active;
		RenderTexture.active = input;
		output.ReadPixels(new Rect(inputOffset.y, inputOffset.x, size.y, size.x), outputOffset.y, outputOffset.x);
		RenderTexture.active = active;
		return output;
	}

	// tensor creation
	HashSet<Texture> texSet = new HashSet<Texture>();
	Dictionary<string,RenderTexture> rtDict = new Dictionary<string,RenderTexture>();
	public virtual RenderTexture PersistentGPUTensor(string name, int size0, int size1, VertexAttributeFormat dtype=VertexAttributeFormat.Float32, int mipmap=0) {
		if(rtDict.ContainsKey(name)) {
			var rt = rtDict[name];
			Debug.Assert(size0 == Size0(rt) && size1 == Size1(rt) && dtype == DType(rt) && mipmap == Mipmap(rt));
			return rt;
		}
		var tex = RenderTexture.GetTemporary(GPUTensorDescriptor(size0, size1, dtype:dtype, mipmap:mipmap));
		Debug.Assert(!texSet.Contains(tex));
		rtDict[name] = tex;
		return tex;
	}
	public virtual RenderTexture GPUTensor(int size0, int size1, VertexAttributeFormat dtype=VertexAttributeFormat.Float32, int mipmap=0) {
		var tex = RenderTexture.GetTemporary(GPUTensorDescriptor(size0, size1, dtype:dtype, mipmap:mipmap));
		Debug.Assert(!texSet.Contains(tex));
		texSet.Add(tex);
		return tex;
	}
	public virtual Texture2D CPUTensor(int size0, int size1, int size2=4, VertexAttributeFormat dtype=VertexAttributeFormat.Float32) {
		var (width, height, textureFormat) = CPUTensorDescriptor(size0, size1, dtype:dtype);
		var tex = new Texture2D(width, height, textureFormat, mipChain:false, linear:true);
		Debug.Assert(!texSet.Contains(tex));
		texSet.Add(tex);
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
		var clone = CPUTensor(Size0(input), Size1(input), dtype:DType(input));
		Copy(clone, (RenderTexture)input, new Vector2Int(Size0(input), Size1(input)));
		var output = GetData(clone).ToArray();
		Release(clone);
		return output;
	}
	public void DebugTensor(Texture2D input) {
		var data = GetData(input);
		DebugTensor(data, Size0(input), Size1(input)*4);
	}
	public void DebugTensor(RenderTexture input) {
		var data = new NativeArray<float>(GetData(input), Allocator.Temp);
		DebugTensor(data, Size0(input), Size1(input)*4);
		data.Dispose();
	}
	void DebugTensor(NativeArray<float> data, int row, int col) {
		Debug.Log($"shape: {row}, {col}");
		for(int i=0; i<row; i++)
			Debug.Log(string.Join(", ", new NativeSlice<float>(data, i*col, col).Select(x => x.ToString("F4"))));
	}

	protected (int,int,TextureFormat) CPUTensorDescriptor(int size0, int size1, int size2=4, VertexAttributeFormat dtype=VertexAttributeFormat.Float32) {
		return (size1, size0, dtypeToTexFormat[(dtype,size2)]);
	}
	protected RenderTextureDescriptor GPUTensorDescriptor(int size0, int size1, int size2=4, VertexAttributeFormat dtype=VertexAttributeFormat.Float32, int mipmap=0) {
		var desc = new RenderTextureDescriptor(size1, size0, dtypeToRTFormat[(dtype,size2)], 0);
		while(mipmap > 0 && ((desc.width << mipmap) > 16384 || (desc.height << mipmap) > 16384))
			mipmap --;
		if(mipmap > 0) {
			desc.width <<= mipmap;
			desc.height <<= mipmap;
			desc.useMipMap = true;
			desc.autoGenerateMips = true;
			desc.mipCount = 1+mipmap;
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