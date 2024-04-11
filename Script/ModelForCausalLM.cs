using UnityEngine;

namespace ShaderGPT {
public abstract class ModelForCausalLM : Module {
	public int maxLength = 2048;
	public abstract (Texture, Texture) ForCausalLM(Texture input_ids);
	public ModelForCausalLM(TensorNN nn, TextAsset configJson): base(nn) {}

	[System.Serializable]
	class Config {
		public string model_type;
	}
	static ModelForCausalLM FromPretrained(TensorNN nn, TextAsset configJson) {
		var config = JsonUtility.FromJson<Config>(configJson.text);
		switch(config.model_type) {
		case "gpt2":
			return new Models.GPT2(nn, configJson);
		case "gpt_neo":
			return new Models.GPTNeo(nn, configJson);
		case "gpt_neox":
			return new Models.GPTNeoX(nn, configJson);
		case "llama":
		case "mistral":
		case "qwen2":
			return new Models.Llama(nn, configJson);
		case "phi":
			return new Models.Phi(nn, configJson);
		default:
			Debug.LogError($"unsupported architecture {config.model_type}");
			return null;
		}
	}
	public static ModelForCausalLM FromPretrained(TensorNN nn, TextAsset configJson, Texture[] textures) {
		var model = FromPretrained(nn, configJson);
		model.LoadStateDict(textures);
		return model;
	}
}
}