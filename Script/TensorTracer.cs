using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace ShaderGPT {
public class TensorTracer: TensorContext {
	Dictionary<int, RenderTextureDescriptor> rtDesc = new Dictionary<int, RenderTextureDescriptor>();
	Dictionary<int, (int,int,TextureFormat)> texDesc = new Dictionary<int, (int,int,TextureFormat)>();
	List<Material> matList = new List<Material>();
	List<int> matSplits = new List<int>();
	public override RenderTexture GPUTensor(int size0, int size1, VertexAttributeFormat? dtype=null, int lod=0, bool autoGenMips=true) {
		var tex = RenderTexture.GetTemporary(GPUTensorDescriptor(size0, size1, dtype:dtype, lod:lod, autoGenMips:autoGenMips));
		rtDesc[tex.GetInstanceID()] = tex.descriptor;
		Debug.Assert(!texSet.Contains(tex));
		texSet.Add(tex);
		FixSize0(tex, size0);
		return tex;
	}
	public override Texture2D CPUTensor(int size0, int size1, VertexAttributeFormat? dtype=null) {
		var (width, height, textureFormat) = CPUTensorDescriptor(size0, size1, dtype:dtype);
		var tex = new Texture2D(width, height, textureFormat, mipChain:false, linear:true);
		texDesc[tex.GetInstanceID()] = (width, height, textureFormat);
		Debug.Assert(!texSet.Contains(tex));
		texSet.Add(tex);
		FixSize0(tex, size0);
		return tex;
	}
	public override void Blit(RenderTexture rt, Material mat) {
		Graphics.Blit(null, rt, mat, 0);
		matList.Add(mat);
	}
	public void Split() {
		matSplits.Add(matList.Count);
	}
	public Object Export(string path) {
		var names = new HashSet<string>();
		foreach(var shader in matList.Select(mat => mat.shader).Distinct())
			for(int i=0; i<shader.GetPropertyCount(); i++)
				if(shader.GetPropertyType(i) == ShaderPropertyType.Texture)
					names.Add(shader.GetPropertyName(i));
#if UNITY_EDITOR
		var remap = new Dictionary<int,Texture>();
		var go = new GameObject("trace");
		var prefab = PrefabUtility.SaveAsPrefabAssetAndConnect(go, path, InteractionMode.AutomatedAction);
		// remove old textures and materials
		foreach(var o in AssetDatabase.LoadAllAssetsAtPath(path))
			if(o is Texture || o is Material)
				Object.DestroyImmediate(o, true);
		{
			// NOTE: mipMapBias isn't recorded
			var idx = 0;
			foreach(var pair in rtDesc) {
				var tex = new RenderTexture(pair.Value);
				tex.name = $"rt{idx++}";
				AssetDatabase.AddObjectToAsset(tex, path);
				remap[pair.Key] = tex;
			}
			idx = 0;
			foreach(var pair in texDesc) {
				var (width, height, format) = pair.Value;
				var tex = new Texture2D(width, height, textureFormat:format, mipChain:false, linear:true);
				tex.name = $"tex{idx++}";
				AssetDatabase.AddObjectToAsset(tex, path);
				remap[pair.Key] = tex;
			}
			idx = 0;
			foreach(var mat in matList) {
				foreach(var name in names) {
					if(!mat.HasProperty(name))
						continue;
					var tex = mat.GetTexture(name);
					if(object.ReferenceEquals(tex, null))
						continue;
					var id = tex.GetInstanceID();
					if(remap.ContainsKey(id))
						mat.SetTexture(name, remap[id]);
				}
				mat.name = $"mat{idx++}";
				AssetDatabase.AddObjectToAsset(mat, path);
			}
		}
		for(int i=0; i<=matSplits.Count; i++) {
			var start = i>0? matSplits[i-1] : 0;
			var stop = i<matSplits.Count ? matSplits[i] : matList.Count;
			var arr = new Material[stop-start];
			for(int j=start; j<stop; j++)
				arr[j-start] = matList[j];
			var child = new GameObject($"group{i}", typeof(MeshRenderer));
			child.transform.parent = go.transform;
			child.GetComponent<MeshRenderer>().sharedMaterials = arr;
		}
		PrefabUtility.ApplyPrefabInstance(go, InteractionMode.AutomatedAction);
		Object.Destroy(go);
		AssetDatabase.SaveAssets();
		return prefab;
#else
		return null;
#endif
	}
}
}