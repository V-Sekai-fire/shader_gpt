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
	
	static MonoBehaviour ImportModel(string folder) {
		var configJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "config.json"));
		var tokenizerJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "tokenizer.json"));
		var testcaseJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Path.Join(folder, "testcase.json"));
		var config = JsonUtility.FromJson<Config>(configJson.text);
		var model_type = config.model_type;
		Debug.Log($"{folder} : {model_type}");

		var texturePaths = GetTexturePaths(folder);
		AssetDatabase.StartAssetEditing();
		try {
			foreach(var path in texturePaths)
				FixTextureImport(path);
		} finally {
			AssetDatabase.StopAssetEditing();
		}
		
		if(model_type == "vits") {
			var go = new GameObject(Path.GetFileName(folder), typeof(BasicTTS));
			var vits = go.GetComponent<BasicTTS>();
			vits.shaders = GetShaders();
			vits.textures = texturePaths.Select(path => AssetDatabase.LoadAssetAtPath<Texture>(path)).ToArray();
			vits.configJson = configJson;
			vits.tokenizerJson = tokenizerJson;
			vits.testcaseJson = testcaseJson;
			return vits;
		} else {
			var go = new GameObject(Path.GetFileName(folder), typeof(BasicLM));
			var lm = go.GetComponent<BasicLM>();
			lm.shaders = GetShaders();
			lm.textures = texturePaths.Select(path => AssetDatabase.LoadAssetAtPath<Texture>(path)).ToArray();
			lm.configJson = configJson;
			lm.tokenizerJson = tokenizerJson;
			lm.testcaseJson = testcaseJson;
			return lm;
		}
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
		if(path.EndsWith(".q8.png"))
			return TextureImporterFormat.RGBA32;
		else if(path.EndsWith(".q8.exr") || path.EndsWith(".q8.idx.exr"))
			return TextureImporterFormat.RGBAFloat;
		else if(File.Exists(Path.ChangeExtension(path, ".q8.png")) || File.Exists(Path.ChangeExtension(path, ".q8.exr"))) {
			if(dtype == DataType.Unorm4)
				return path.EndsWith(".png") ? TextureImporterFormat.ARGB16 : TextureImporterFormat.RGBA32;
			else if(dtype == DataType.Unorm8)
				return TextureImporterFormat.RGBA32;
		}
		if(path.EndsWith(".png"))
			return TextureImporterFormat.RGBA32;
		return dtype == DataType.Float16 ? TextureImporterFormat.RGBAHalf : TextureImporterFormat.RGBAFloat;
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