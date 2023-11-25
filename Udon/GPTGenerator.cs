using UnityEngine;
using UnityEngine.Rendering;
#if UDON
using VRC.SDK3.Rendering;
using Graphics = VRC.SDKBase.VRCGraphics;
#endif

namespace ShaderGPT.Udon {
#if UDON
[UdonSharp.UdonBehaviourSyncMode(UdonSharp.BehaviourSyncMode.None)]
public class GPTGenerator : UdonSharp.UdonSharpBehaviour
#else
public class GPTGenerator : MonoBehaviour
#endif
{
	[Header("Model")]
	public GameObject modelPrefab;

	[Header("Config")]
	public int maxLength = 2048;
	public float temperature = 0;
	public int frameStep = 1;
#if UDON
	public VRC.Udon.UdonBehaviour callback;
#else
	public MonoBehaviour callback;
#endif

	private Material[] materials;
	private Material[] matDynRangeMask;
	private Material[] matDynOutputOff;
	private Material[] matDynRotaryOff;
	private Material matGumbel;
	private Material matOutput;
	private RenderTexture bufOutput;
	void LoadModel() {
		var renderer = modelPrefab.GetComponentInChildren<MeshRenderer>();
		// if renderer is instanced, use ".materials" to avoid saving modification
		materials = renderer.GetInstanceID() < 0 ? renderer.materials : renderer.sharedMaterials;

		var maxPosLength = 0;
		foreach(var mat in materials) {
			var rt = (RenderTexture)mat.GetTexture("_OutputTex");
			maxPosLength = Mathf.Max(maxPosLength, rt.height);
		}
		maxLength = Mathf.Min(maxLength, maxPosLength);

		matOutput = materials[materials.Length-1];
		bufOutput = (RenderTexture)matOutput.GetTexture("_OutputTex");
		matDynRangeMask = new Material[materials.Length];
		matDynOutputOff = new Material[materials.Length];
		matDynRotaryOff = new Material[materials.Length];
		matGumbel = null;
		var matDynRangeMaskLength = 0;
		var matDynOutputOffLength = 0;
		var matDynRotaryOffLength = 0;
		foreach(var mat in materials) {
			var rt = (RenderTexture)mat.GetTexture("_OutputTex");
			if(mat.HasProperty("_RangeMask") && mat.GetVector("_RangeMask").x == 1)
				matDynRangeMask[matDynRangeMaskLength++] = mat;
			if(mat.HasProperty("_OutputOff") && rt.height == maxPosLength)
				matDynOutputOff[matDynOutputOffLength++] = mat;
			if(mat.shaderKeywords.Length >= 1 && mat.shaderKeywords[0] == "FUNC_ROTARY")
				matDynRotaryOff[matDynRotaryOffLength++] = mat;
			if(mat.shaderKeywords.Length >= 1 && mat.shaderKeywords[0] == "FUNC_GUMBEL")
				matGumbel = mat;
		}
		matDynRangeMask = Resize(matDynRangeMask, matDynRangeMaskLength);
		matDynOutputOff = Resize(matDynOutputOff, matDynOutputOffLength);
		matDynRotaryOff = Resize(matDynRotaryOff, matDynRotaryOffLength);
	}

	private int tokenIndex;
	[System.NonSerialized] public int[] inputTokens;
	[System.NonSerialized] public int outputToken;
	void RunModel() {
		var deltaRangeMask = new Vector4(0, 0, tokenIndex-1, tokenIndex-1);
		var deltaOutputOff = new Vector4(tokenIndex-1, 0, 0, 0);
		var deltaRotaryOff = deltaOutputOff;
		foreach(var mat in matDynRangeMask)
			mat.SetVector("_RangeMask", mat.GetVector("_RangeMask") + deltaRangeMask);
		foreach(var mat in matDynOutputOff)
			mat.SetVector("_OutputOff", mat.GetVector("_OutputOff") + deltaOutputOff);
		foreach(var mat in matDynRotaryOff)
			mat.SetVector("_RotaryOff", mat.GetVector("_RotaryOff") + deltaRotaryOff);

		matGumbel.SetVector("_Weight", Vector4.one*temperature);
		if(tokenIndex < inputTokens.Length) {
			matOutput.SetVector("_Weight", Vector4.zero);
			matOutput.SetVector("_Bias", new Vector4(inputTokens[tokenIndex],tokenIndex,0,0));
		} else {
			matOutput.SetVector("_Weight", Vector4.one);
			matOutput.SetVector("_Bias", new Vector4(0,tokenIndex,0,0));
		}
		foreach(var mat in materials)
			Graphics.Blit(null, (RenderTexture)mat.GetTexture("_OutputTex"), mat, 0);

		foreach(var mat in matDynRangeMask)
			mat.SetVector("_RangeMask", mat.GetVector("_RangeMask") - deltaRangeMask);
		foreach(var mat in matDynOutputOff)
			mat.SetVector("_OutputOff", mat.GetVector("_OutputOff") - deltaOutputOff);
		foreach(var mat in matDynRotaryOff)
			mat.SetVector("_RotaryOff", mat.GetVector("_RotaryOff") - deltaRotaryOff);
		tokenIndex++;
	}

	private int frameIndex;
	public void OnEnable() {
		if(!matOutput)
			LoadModel();
		tokenIndex = 0;
		frameIndex = 0;
	}
	public void Update() {
		if(frameIndex != 0) {
			frameIndex--;
			return;
		}
		frameIndex = frameStep;
		if(tokenIndex >= maxLength) {
			this.enabled = false;
			return;
		}
		RunModel();
#if UDON
		VRCAsyncGPUReadback.Request(bufOutput, 0, TextureFormat.RGBAFloat, (VRC.Udon.Common.Interfaces.IUdonEventReceiver)(Component)this);
#else
		AsyncGPUReadback.Request(bufOutput, 0, TextureFormat.RGBAFloat, OnAsyncGpuReadbackComplete);
#endif
	}
	private Color[] readbackData = new Color[1];
#if UDON
	public override void OnAsyncGpuReadbackComplete(VRCAsyncGPUReadbackRequest request)
#else
	public void OnAsyncGpuReadbackComplete(AsyncGPUReadbackRequest request)
#endif
	{
		if (request.hasError || !request.done) return;
		if (!this) return;
#if UDON
		request.TryGetData(readbackData);
#else
		request.GetData<Color>().CopyTo(readbackData);
#endif
		var index = Mathf.RoundToInt(readbackData[0].g);
		if(index <= tokenIndex) { // race condition
			outputToken = Mathf.RoundToInt(readbackData[0].r);
#if UDON
			callback.SendCustomEvent("OnOutputToken");
#else
			callback.SendMessage("OnOutputToken");
#endif
		}
	}

	static Material[] Resize(Material[] src, int n) {
		var dst = new Material[n];
		System.Array.Copy(src, dst, n);
		return dst;
	}
}
}