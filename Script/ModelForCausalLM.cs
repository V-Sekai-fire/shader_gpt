using UnityEngine;

namespace ShaderGPT {
[System.Serializable]
public class ModelForCausalLMConfig {
	public string model_type;
	public int vocab_size;
}
public abstract class ModelForCausalLM : Module {
	public int max_length = 2048; // generation config
	public abstract (Texture, Texture) ForCausalLM(Texture input_ids);
	public ModelForCausalLM(TensorNN nn): base(nn) {}

	static ModelForCausalLM FromPretrained(TensorNN nn, TextAsset configJson) {
		var config = JsonUtility.FromJson<ModelForCausalLMConfig>(configJson.text);
		switch(config.model_type) {
		case "gpt2":
			return new Models.GPT2(nn, configJson);
		case "gpt_neo":
			return new Models.GPTNeo(nn, configJson);
		case "gpt_neox":
			return new Models.GPTNeoX(nn, configJson);
		case "gemma":
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
public abstract class ModelForCausalLM<T> : ModelForCausalLM where T : ModelForCausalLMConfig {
	public T config;
	public ModelForCausalLM(TensorNN nn, TextAsset configJson): base(nn) {
		config = JsonUtility.FromJson<T>(configJson.text);
	}
}
}