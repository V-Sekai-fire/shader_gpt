#if UNITY_EDITOR
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEditor;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace ShaderGPT {
public class ModelImporter {
	[MenuItem("Assets/ShaderGPT/ImportModel", false, 100)]
	static void ImportModel() {
		foreach(var o in Selection.objects) {
			var path = AssetDatabase.GetAssetPath(o);
			if(Directory.Exists(path))
				EditorGUIUtility.PingObject(ImportModel(path));
		}
	}
	[MenuItem("Assets/ShaderGPT/ProfileModel", false, 101)]
	static void ProfileModel() {
		foreach(var o in Selection.objects) {
			var path = AssetDatabase.GetAssetPath(o);
			if(Directory.Exists(path))
				ProfileModel(path);
		}
	}
	[MenuItem("Assets/ShaderGPT/UseFloat32", false, 200)]
	static void UseFloat32() {
		foreach(var o in Selection.objects) {
			var path = AssetDatabase.GetAssetPath(o);
			if(Directory.Exists(path))
				SetModelCompression(path, DataType.Float32);
		}
	}
	[MenuItem("Assets/ShaderGPT/UseFloat16", false, 201)]
	static void UseFloat16() {
		foreach(var o in Selection.objects) {
			var path = AssetDatabase.GetAssetPath(o);
			if(Directory.Exists(path))
				SetModelCompression(path, DataType.Float16);
		}
	}
	[MenuItem("Assets/ShaderGPT/UseUnorm8", false, 202)]
	static void UseUnorm8() {
		foreach(var o in Selection.objects) {
			var path = AssetDatabase.GetAssetPath(o);
			if(Directory.Exists(path))
				SetModelCompression(path, DataType.Unorm8);
		}
	}
	[MenuItem("Assets/ShaderGPT/UseUnorm4", false, 203)]
	static void UseUnorm4() {
		foreach(var o in Selection.objects) {
			var path = AssetDatabase.GetAssetPath(o);
			if(Directory.Exists(path))
				SetModelCompression(path, DataType.Unorm4);
		}
	}
	
	static GPTBase ImportModel(string folder) {
		var configJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "config.json"));
		var tokenizerJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "tokenizer.json"));
		var testcaseJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "testcase.json"));
		var config = JsonUtility.FromJson<Config>(configJson.text);
		var model_type = config.model_type;
		Debug.Log($"{folder} : {model_type}");

		var type = default(System.Type);
		if(model_type == "gpt2")
			type = typeof(GPT2);
		else if(model_type == "gpt_neo")
			type = typeof(GPTNeo);
		else if(model_type == "gpt_neox")
			type = typeof(GPTNeoX);
		else if(model_type == "phi-msft")
			type = typeof(Phi);
		else {
			Debug.LogError($"unsupported architecture {model_type}");
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
	static void SetModelCompression(string folder, DataType dtype) {
		var texturePaths = GetTexturePaths(folder);
		AssetDatabase.StartAssetEditing();
		try {
			foreach(var path in texturePaths)
				SetTextureFormat(path, CalcCompressionFormat(path, dtype));
		} finally {
			AssetDatabase.StopAssetEditing();
		}
	}
	static void ProfileModel(string folder) {
		var texturePaths = GetTexturePaths(folder);
		var groupSizes = new Dictionary<string, long>();
		long totalSize = 0;
		foreach(var path in texturePaths) {
			var group = string.Join(".", Path.GetFileNameWithoutExtension(path).Split(".").TakeLast(2));
			var tex = AssetDatabase.LoadAssetAtPath<Texture>(path);
			var size = (long)GraphicsFormatUtility.ComputeMipmapSize(tex.width, tex.height, tex.graphicsFormat);
			totalSize += size;
			groupSizes.TryGetValue(group, out var groupSize);
			groupSizes[group] = groupSize + size;
		}
		var sb = new System.Text.StringBuilder();
		sb.AppendLine($"TOTAL: {totalSize>>20} MB");
		foreach(var pair in groupSizes.OrderBy(p => -p.Value))
			if((pair.Value>>20) > 0)
				sb.AppendLine($"{pair.Key}: {pair.Value>>20} MB");
		Debug.Log(sb.ToString());
	}

	static void FixTextureImport(string path) {
		var importer = (TextureImporter)AssetImporter.GetAtPath(path);
		if(importer.sRGBTexture == false && importer.npotScale == TextureImporterNPOTScale.None && importer.maxTextureSize == 16384)
			return;

		importer.sRGBTexture = false;
		importer.npotScale = TextureImporterNPOTScale.None;
		importer.mipmapEnabled = false;

		var fmt32 = CalcCompressionFormat(path, DataType.Float32);
		var fmt16 = CalcCompressionFormat(path, DataType.Float16);
		var settings = importer.GetDefaultPlatformTextureSettings();
		settings.maxTextureSize = 16384;
		if(settings.format != fmt32 && settings.format != fmt16)
			settings.format = fmt32;
		importer.SetPlatformTextureSettings(settings);
		importer.SaveAndReimport();
	}
	static void SetTextureFormat(string path, TextureImporterFormat format) {
		var importer = (TextureImporter)AssetImporter.GetAtPath(path);
		var settings = importer.GetDefaultPlatformTextureSettings();
		var standalone = importer.GetPlatformTextureSettings("Standalone");
		if(settings.format == format && !(format == TextureImporterFormat.RGBAFloat && standalone.overridden))
			return;

		settings.format = format;
		importer.SetPlatformTextureSettings(settings);
		if(format == TextureImporterFormat.RGBAFloat) {
			standalone.overridden = false;
			importer.SetPlatformTextureSettings(standalone);
		}
		importer.SaveAndReimport();
	}
	static TextureImporterFormat CalcCompressionFormat(string path, DataType dtype) {
		const int minSizeQ4 = 16*1024*1024;
		return path.EndsWith(".png") ? TextureImporterFormat.RGBA32 :
			dtype == DataType.Float32 ? TextureImporterFormat.RGBAFloat :
			!File.Exists(Path.ChangeExtension(path, ".q8.png")) ? TextureImporterFormat.RGBAHalf :
			dtype == DataType.Unorm4 ? (
				new FileInfo(path).Length > minSizeQ4 ? TextureImporterFormat.ARGB16 : TextureImporterFormat.RGBA32) :
			dtype == DataType.Unorm8 ? TextureImporterFormat.RGBA32 :
			TextureImporterFormat.RGBAHalf;
	}
	static string[] GetTexturePaths(string folder) {
		return AssetDatabase.FindAssets("t:Texture", new string[]{folder}).Select(
			guid => AssetDatabase.GUIDToAssetPath(guid)).ToArray();
	}
	static Shader[] GetShaders() {
		return ShaderUtil.GetAllShaderInfo().Where(info => info.name.StartsWith("GPT/")).Select(
			info => Shader.Find(info.name)).ToArray();
	}

	enum DataType {
		Float32,
		Float16,
		Unorm8,
		Unorm4,
	}
	[System.Serializable]
	class Config {
		public string model_type;
	}
}
}
#endif