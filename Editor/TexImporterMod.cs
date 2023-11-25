#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

namespace ShaderGPT {
public class TexImporterMod {
	[MenuItem("CONTEXT/TextureImporter/ImportAsTensor")]
	static void ImportAsTensor(MenuCommand command) {
		var importer = (TextureImporter)command.context;
		importer.sRGBTexture = false;
		importer.npotScale = TextureImporterNPOTScale.None;
		importer.mipmapEnabled = false;

		var settings = importer.GetPlatformTextureSettings("Standalone");
		settings.maxTextureSize = 16384;
		settings.format = TextureImporterFormat.RGBAFloat;
		settings.overridden = true;
		importer.SetPlatformTextureSettings(settings);
		
		importer.SaveAndReimport();
	}
}
}
#endif