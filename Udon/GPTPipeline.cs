using UnityEngine;

namespace ShaderGPT.Udon {
#if UDON
[UdonSharp.UdonBehaviourSyncMode(UdonSharp.BehaviourSyncMode.None)]
public class GPTPipeline : UdonSharp.UdonSharpBehaviour
#else
public class GPTPipeline : MonoBehaviour
#endif
{
	public GPTTokenizer tokenizer;
	public GPTGenerator generator;

	public UnityEngine.UI.Text inputText;
	public UnityEngine.UI.Text outputText;
	public UnityEngine.UI.Slider temperatureSlider;

	private bool eos;
	public void OnEnable() {
#if UDON
		generator.eventTarget = (VRC.Udon.UdonBehaviour)(Component)this;
#else
		generator.eventTarget = this;
#endif
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
		if(token == tokenizer.eos_token_id) {
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