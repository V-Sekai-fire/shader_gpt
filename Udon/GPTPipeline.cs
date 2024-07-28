using UnityEngine;

namespace ShaderGPT.Udon {
public class GPTPipeline : UdonMonoBehaviour {
	public GPTTokenizer tokenizer;
	public GPTGenerator generator;

	public UnityEngine.UI.Text inputText;
	public UnityEngine.UI.Text outputText;
	public UnityEngine.UI.Slider temperatureSlider;

	public bool ignoreEosToken;

	private bool eos;
	public void OnEnable() {
		generator.eventTarget = this.AsUdonBehaviour();
		generator.eventMethod = nameof(OnOutputToken);
		
		generator.inputTokens = tokenizer.Encode(inputText.text);
		generator.inputIndex = 0;
		tokenizer.decodeState = 0;
		outputText.text = "";
		eos = false;
		generator.enabled = true;
	}
	public void OnDisable() {
		if(!generator)
			return;
		generator.enabled = false;
	}
	public void OnOutputToken() {
		if(eos)
			return;
		if(!generator)
			return;
		var token = generator.outputToken;
		var index = generator.outputIndex;
		if(token == tokenizer.eos_token_id && index >= generator.inputTokens.Length && !ignoreEosToken) {
			eos = true;
			generator.enabled = false;
			return;
		}
		outputText.text += tokenizer.Decode(token);
	}
	public void UpdateConfig() {
		generator.temperature = temperatureSlider.value / temperatureSlider.maxValue;
	}
}
}