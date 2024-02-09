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
	Dictionary<string,RenderTexture> persistDict = new Dictionary<string,RenderTexture>();
	Dictionary<int, RenderTextureDescriptor> rtDesc = new Dictionary<int, RenderTextureDescriptor>();
	Dictionary<int, (int,int,TextureFormat)> texDesc = new Dictionary<int, (int,int,TextureFormat)>();
	List<Material> matList = new List<Material>();
	public override RenderTexture PersistentGPUTensor(string name, int size0, int size1, VertexAttributeFormat dtype=VertexAttributeFormat.Float32, int mipmap=0) {
		if(persistDict.ContainsKey(name))
			return persistDict[name];
		var tex = new RenderTexture(GPUTensorDescriptor(size0, size1, dtype:dtype, mipmap:mipmap));
		rtDesc[tex.GetInstanceID()] = tex.descriptor;
		persistDict[name] = tex;
		return tex;
	}
	public override RenderTexture GPUTensor(int size0, int size1, VertexAttributeFormat dtype=VertexAttributeFormat.Float32, int mipmap=0, bool autoMips=true) {
		var tex = RenderTexture.GetTemporary(GPUTensorDescriptor(size0, size1, dtype:dtype, mipmap:mipmap, autoMips:autoMips));
		rtDesc[tex.GetInstanceID()] = tex.descriptor;
		return tex;
	}
	public override Texture2D CPUTensor(int size0, int size1, int size2=4, VertexAttributeFormat dtype=VertexAttributeFormat.Float32) {
		var (width, height, textureFormat) = CPUTensorDescriptor(size0, size1, dtype:dtype);
		var tex = new Texture2D(width, height, textureFormat, mipChain:false, linear:true);
		texDesc[tex.GetInstanceID()] = (width, height, textureFormat);
		return tex;
	}
	public override void Release(Texture tex) {
		var rt = tex as RenderTexture;
		if(rt)
			RenderTexture.ReleaseTemporary(rt);
		else
			Object.Destroy(tex);
	}
	public override void ReleasePersistent() {
		foreach(var pair in persistDict)
			Object.Destroy(pair.Value);
		persistDict.Clear();
	}
	public override void Blit(RenderTexture rt, Material mat) {
		Graphics.Blit(null, rt, mat, 0);
		matList.Add(mat);
	}
	public Object Export(string path) {
		var names = new HashSet<string>();
		foreach(var shader in matList.Select(mat => mat.shader).Distinct())
			for(int i=0; i<shader.GetPropertyCount(); i++)
				if(shader.GetPropertyType(i) == ShaderPropertyType.Texture)
					names.Add(shader.GetPropertyName(i));
#if UNITY_EDITOR
		var remap = new Dictionary<int,Texture>();
		var go = new GameObject("foo");
		var child = new GameObject("bar", typeof(MeshRenderer));
		child.transform.parent = go.transform; // putting renderer at root may cause editor lag
		var prefab = PrefabUtility.SaveAsPrefabAssetAndConnect(go, path, InteractionMode.AutomatedAction);
		// remove old textures and materials
		foreach(var o in AssetDatabase.LoadAllAssetsAtPath(path))
			if(o is Texture || o is Material)
				Object.DestroyImmediate(o, true);
		{
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
		child.GetComponent<MeshRenderer>().sharedMaterials = matList.ToArray();
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