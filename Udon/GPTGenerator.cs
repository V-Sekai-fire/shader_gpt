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
	public float repetitionPenalty = 1f;
	public int frameStep = 1;
#if UDON
	public VRC.Udon.UdonBehaviour eventTarget;
#else
	public MonoBehaviour eventTarget;
#endif
	public string eventMethod;

	private Material[] materials;
	private Material matRepeat;
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
		matRepeat = null;
		matGumbel = null;
		foreach(var mat in materials) {
			var rt = (RenderTexture)mat.GetTexture("_OutputTex");
			if(mat.shaderKeywords.Length == 1) {
				var keyword = mat.shaderKeywords[0];
				if(keyword == "FUNC_RELU")
					matRepeat = mat;
				if(keyword == "FUNC_GUMBEL")
					matGumbel = mat;
			}
		}
	}

	// you need to set inputIndex and inputTokens before enabling
	[System.NonSerialized] public int inputIndex;
	[System.NonSerialized] public int[] inputTokens;
	[System.NonSerialized] public int outputIndex;
	[System.NonSerialized] public int outputToken;
	private RenderTexture[] deferList = new RenderTexture[16];
	void GenerateToken() {
		// make sure the buffer stores the last token
		if(0 < inputIndex && inputIndex <= inputTokens.Length) {
			matOutput.SetVector("_Mul", Vector4.zero);
			matOutput.SetVector("_Add", new Vector4(inputTokens[inputIndex-1],inputIndex-1,0,0));
			Graphics.Blit(null, bufOutput, matOutput, 0);
		}

		matRepeat.SetFloat("_Eps", repetitionPenalty*repetitionPenalty);
		matRepeat.SetVector("_Mul", Vector4.one/repetitionPenalty);
		matGumbel.SetVector("_Mul", Vector4.one*temperature);
		if(inputIndex < inputTokens.Length) {
			matOutput.SetVector("_Mul", Vector4.zero);
			matOutput.SetVector("_Add", new Vector4(inputTokens[inputIndex],inputIndex,0,0));
		} else {
			matOutput.SetVector("_Mul", Vector4.one);
			matOutput.SetVector("_Add", new Vector4(0,inputIndex,0,0));
		}
		var deferCount = 0;
		foreach(var mat in materials) {
			var rt = (RenderTexture)mat.GetTexture("_OutputTex");
			// merge GenerateMips to reduce PS/CS switch
			if(rt.useMipMap && !rt.autoGenerateMips)
				deferList[deferCount++] = rt;
			else
				while(deferCount > 0)
					deferList[--deferCount].GenerateMips();
			Graphics.Blit(null, rt, mat, 0);
		}

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
}
}