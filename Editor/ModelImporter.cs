#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Linq;

namespace ShaderGPT {
public class ModelImporter {
	[MenuItem("Assets/ImportGPTModel")]
	static void ImportGPTModel() {
		var path = AssetDatabase.GetAssetPath(Selection.activeObject);
		if(Directory.Exists(path))
			EditorGUIUtility.PingObject(ImportGPTModel(path));
	}
	
	static GPTBase ImportGPTModel(string folder) {
		var configJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "config.json"));
		var tokenizerJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "tokenizer.json"));
		var testcaseJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "testcase.json"));
		var config = JsonUtility.FromJson<Config>(configJson.text);
		var arch = config.architectures[0];
		Debug.Log(arch);

		var type = default(System.Type);
		if(arch == "GPT2LMHeadModel")
			type = typeof(GPT2);
		else if(arch == "GPTNeoForCausalLM")
			type = typeof(GPTNeo);
		else if(arch == "GPTNeoXForCausalLM")
			type = typeof(GPTNeoX);
		else if(arch == "PhiForCausalLM")
			type = typeof(Phi);
		else {
			Debug.LogError($"unsupported architecture {arch}");
			return null;
		}

		var texturePaths = GetTexturePaths(folder);
		AssetDatabase.StartAssetEditing();
		try {
			foreach(var path in texturePaths)
				FixTextureImport(path);
		} finally {
			AssetDatabase.StopAssetEditing();
		}
		
		var go = new GameObject(Path.GetFileName(folder), type);
		var gpt = go.GetComponent<GPTBase>();
		gpt.shaders = GetShaders();
		gpt.textures = texturePaths.Select(path => AssetDatabase.LoadAssetAtPath<Texture>(path)).ToArray();
		gpt.configJson = configJson;
		gpt.tokenizerJson = tokenizerJson;
		gpt.testcaseJson = testcaseJson;
		return gpt;
	}

	static void FixTextureImport(string path) {
		var importer = (TextureImporter)AssetImporter.GetAtPath(path);
		if(importer.sRGBTexture == false && importer.npotScale == TextureImporterNPOTScale.None && importer.maxTextureSize == 16384)
			return;

		importer.sRGBTexture = false;
		importer.npotScale = TextureImporterNPOTScale.None;
		importer.mipmapEnabled = false;

		// importer.ClearPlatformTextureSettings("Standalone");
		var settings = importer.GetDefaultPlatformTextureSettings();
		settings.maxTextureSize = 16384;
		settings.format = TextureImporterFormat.RGBAFloat;
		importer.SetPlatformTextureSettings(settings);

		importer.SaveAndReimport();
	}
	static string[] GetTexturePaths(string folder) {
		return AssetDatabase.FindAssets("t:Texture", new string[]{folder}).Select(
			guid => AssetDatabase.GUIDToAssetPath(guid)).ToArray();
	}
	static Shader[] GetShaders() {
		return ShaderUtil.GetAllShaderInfo().Where(info => info.name.StartsWith("GPT/")).Select(
			info => Shader.Find(info.name)).ToArray();
	}

	[System.Serializable]
	class Config {
		public string[] architectures;
	}
}
}
#endif