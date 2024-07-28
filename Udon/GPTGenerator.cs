using UnityEngine;
using UnityEngine.Rendering;
#if UDON
using MonoBehaviour = VRC.Udon.Common.Interfaces.IUdonEventReceiver;
using Graphics = VRC.SDKBase.VRCGraphics;
using AsyncGPUReadbackRequest = VRC.SDK3.Rendering.VRCAsyncGPUReadbackRequest;
#endif

namespace ShaderGPT.Udon {
public class GPTGenerator : UdonMonoBehaviour {
	public GameObject modelPrefab;
	public int maxLength = 2048;
	public float temperature = 0;
	public float repetitionPenalty = 1f;
	public int frameStep = 1;
	public MonoBehaviour eventTarget;
	public string eventMethod;

	private Material[] matEncoders;
	private Texture2D texTokens;
	void LoadEncoder() {
		texTokens = (Texture2D)matEncoders[0].GetTexture("_InputTex");
	}

	private Material[] matDecoders;
	private Material matRepeat;
	private Material matGumbel;
	private Material matOutput;
	private RenderTexture bufOutput;
	void LoadDecoder() {
		var maxPosLength = 0;
		foreach(var mat in matDecoders) {
			var rt = (RenderTexture)mat.GetTexture("_OutputTex");
			maxPosLength = Mathf.Max(maxPosLength, rt.height);
		}
		maxLength = Mathf.Min(maxLength, maxPosLength);

		matOutput = matDecoders[matDecoders.Length-1];
		bufOutput = (RenderTexture)matOutput.GetTexture("_OutputTex");
		foreach(var mat in matDecoders) {
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

	[System.NonSerialized] public int[] encoderTokens;
	void RunEncoder() {
		var maxTokens = texTokens.height;
		var numTokens = Mathf.Min(maxTokens, encoderTokens.Length);
		var tokenData = new float[maxTokens*4];
		for(int i=0; i<numTokens; i++) {
			tokenData[i*4+0] = encoderTokens[i];
			tokenData[i*4+1] = i;
			tokenData[i*4+2] = numTokens;
		}
		SetPixelData(texTokens, tokenData);
		texTokens.Apply(false, false);

		deferCount = 0;
		foreach(var mat in matEncoders)
			Blit(mat);
	}
	static void SetPixelData(Texture2D tex, float[] data) {
		var bytes = new byte[data.Length*4];
		System.Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
		tex.LoadRawTextureData(bytes);
	}

	// you need to set inputIndex and inputTokens before enabling
	[System.NonSerialized] public int inputIndex;
	[System.NonSerialized] public int[] inputTokens;
	[System.NonSerialized] public int outputIndex;
	[System.NonSerialized] public int outputToken;
	void GenerateToken() {
		var chan2 = matEncoders == null ? 0 : encoderTokens.Length;
		// make sure the buffer stores the last token
		if(0 < inputIndex && inputIndex <= inputTokens.Length) {
			matOutput.SetVector("_Mul", Vector4.zero);
			matOutput.SetVector("_Add", new Vector4(inputTokens[inputIndex-1],inputIndex-1,chan2,0));
			Graphics.Blit(null, bufOutput, matOutput, 0);
		}

		matRepeat.SetFloat("_Eps", repetitionPenalty*repetitionPenalty);
		matRepeat.SetVector("_Mul", Vector4.one/repetitionPenalty);
		matGumbel.SetVector("_Mul", Vector4.one*temperature);
		if(inputIndex < inputTokens.Length) {
			matOutput.SetVector("_Mul", Vector4.zero);
			matOutput.SetVector("_Add", new Vector4(inputTokens[inputIndex],inputIndex,chan2,0));
		} else {
			matOutput.SetVector("_Mul", Vector4.one);
			matOutput.SetVector("_Add", new Vector4(0,inputIndex,chan2,0));
		}
		deferCount = 0;
		foreach(var mat in matDecoders)
			Blit(mat);
		inputIndex++;
	}
	private RenderTexture[] deferList = new RenderTexture[16];
	private int deferCount;
	void Blit(Material mat) {
		var rt = (RenderTexture)mat.GetTexture("_OutputTex");
		// merge GenerateMips to reduce PS/CS switch
		if(rt.useMipMap && !rt.autoGenerateMips)
			deferList[deferCount++] = rt;
		else
			while(deferCount > 0)
				deferList[--deferCount].GenerateMips();
		Graphics.Blit(null, rt, mat, 0);
	}

	private int frameIndex;
	public void OnEnable() {
		if(matDecoders == null) {
			var renderers = modelPrefab.GetComponentsInChildren<MeshRenderer>();
			Debug.Assert(renderers.Length == 1 || renderers.Length == 2);
			if(renderers.Length == 2) {
				var renderer = renderers[0];
				matEncoders = renderer.GetInstanceID() < 0 ? renderer.materials : renderer.sharedMaterials;
				LoadEncoder();
			}
			if(renderers.Length >= 1) {
				var renderer = renderers[renderers.Length-1];
				// if renderer is instanced, use ".materials" to avoid saving modification
				matDecoders = renderer.GetInstanceID() < 0 ? renderer.materials : renderer.sharedMaterials;
				LoadDecoder();
			}
		}
		if(matEncoders != null)
			RunEncoder();
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
		AsyncGPUReadback_Request(bufOutput, 0, TextureFormat.RGBAFloat);
	}
	void OnGPUReadback(float[] data) {
		outputToken = Mathf.RoundToInt(data[0]);
		outputIndex = Mathf.RoundToInt(data[1]);
		if(outputIndex >= inputIndex)
			return; // skip tokens from last sequence
		eventTarget.SendMessage(eventMethod);
	}

	private float[] readbackData;
	public override void OnAsyncGpuReadbackComplete(AsyncGPUReadbackRequest request) {
		if(!(this && this.enabled))
			return;
		if(request.hasError || !request.done)
			return;
		if(readbackData == null)
			readbackData = new float[request.width*request.height*4];
		request.GetData<float>().CopyTo(readbackData);
		OnGPUReadback(readbackData);
	}
}
}