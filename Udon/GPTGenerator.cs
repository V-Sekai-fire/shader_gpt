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
	public GameObject modelPrefab;
	public int maxLength = 2048;
	public float temperature = 0;
	public int skipLastToken = 0;
	public int frameStep = 1;
#if UDON
	public VRC.Udon.UdonBehaviour eventTarget;
#else
	public MonoBehaviour eventTarget;
#endif
	public string eventMethod;

	private Material[] materials;
	private Material[] matDynRangeMask;
	private Material[] matDynOutputOff;
	private Material matGumbel;
	private Material matSample;
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
		matGumbel = null;
		matSample = null;
		var matDynRangeMaskLength = 0;
		var matDynOutputOffLength = 0;
		foreach(var mat in materials) {
			var rt = (RenderTexture)mat.GetTexture("_OutputTex");
			if(mat.HasProperty("_RangeMask") && mat.GetVector("_RangeMask").x == 1)
				matDynRangeMask[matDynRangeMaskLength++] = mat;
			if(mat.HasProperty("_OutputOff") && rt.height == maxPosLength)
				matDynOutputOff[matDynOutputOffLength++] = mat;
			if(mat.shaderKeywords.Length >= 1 && mat.shaderKeywords[0] == "FUNC_GUMBEL")
				matGumbel = mat;
			if(mat.shaderKeywords.Length >= 1 && mat.shaderKeywords[0] == "REDUCE_MINMAX")
				matSample = mat;
		}
		matDynRangeMask = Take(matDynRangeMask, matDynRangeMaskLength);
		matDynOutputOff = Take(matDynOutputOff, matDynOutputOffLength);
	}

	// you need to set inputIndex and inputTokens before enabling
	[System.NonSerialized] public int inputIndex;
	[System.NonSerialized] public int[] inputTokens;
	[System.NonSerialized] public int outputIndex;
	[System.NonSerialized] public int outputToken;
	void GenerateToken() {
		// make sure the buffer stores the last token
		if(0 < inputIndex && inputIndex <= inputTokens.Length) {
			matOutput.SetVector("_Weight", Vector4.zero);
			matOutput.SetVector("_Bias", new Vector4(inputTokens[inputIndex-1],inputIndex-1,0,0));
			Graphics.Blit(null, bufOutput, matOutput, 0);
		}

		var deltaRangeMaskSample = new Vector4(0, 0, 0, -skipLastToken);
		var deltaRangeMask = new Vector4(0, 0, inputIndex-1, inputIndex-1);
		var deltaOutputOff = new Vector4(inputIndex-1, 0, 0, 0);
		matSample.SetVector("_RangeMask", matSample.GetVector("_RangeMask") + deltaRangeMaskSample);
		foreach(var mat in matDynRangeMask)
			mat.SetVector("_RangeMask", mat.GetVector("_RangeMask") + deltaRangeMask);
		foreach(var mat in matDynOutputOff)
			mat.SetVector("_OutputOff", mat.GetVector("_OutputOff") + deltaOutputOff);

		matGumbel.SetVector("_Weight", Vector4.one*temperature);
		if(inputIndex < inputTokens.Length) {
			matOutput.SetVector("_Weight", Vector4.zero);
			matOutput.SetVector("_Bias", new Vector4(inputTokens[inputIndex],inputIndex,0,0));
		} else {
			matOutput.SetVector("_Weight", Vector4.one);
			matOutput.SetVector("_Bias", new Vector4(0,inputIndex,0,0));
		}
		foreach(var mat in materials)
			Graphics.Blit(null, (RenderTexture)mat.GetTexture("_OutputTex"), mat, 0);

		matSample.SetVector("_RangeMask", matSample.GetVector("_RangeMask") - deltaRangeMaskSample);
		foreach(var mat in matDynRangeMask)
			mat.SetVector("_RangeMask", mat.GetVector("_RangeMask") - deltaRangeMask);
		foreach(var mat in matDynOutputOff)
			mat.SetVector("_OutputOff", mat.GetVector("_OutputOff") - deltaOutputOff);
		inputIndex++;
	}

	private int frameIndex;
	public void OnEnable() {
		if(!matOutput)
			LoadModel();
		frameIndex = 0;
	}
	public void Update() {
		if(frameIndex != 0) {
			frameIndex--;
			return;
		}
		frameIndex = frameStep;
		if(inputIndex >= maxLength) {
			this.enabled = false;
			return;
		}
		GenerateToken();
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
		outputIndex = Mathf.RoundToInt(readbackData[0].g);
		if(outputIndex >= inputIndex)
			return; // skip tokens from last sequence
		outputToken = Mathf.RoundToInt(readbackData[0].r);
#if UDON
		eventTarget.SendCustomEvent(eventMethod);
#else
		eventTarget.SendMessage(eventMethod);
#endif
	}

	static T[] Take<T>(T[] src, int n) {
		var dst = new T[n];
		System.Array.Copy(src, dst, n);
		return dst;
	}
}
}