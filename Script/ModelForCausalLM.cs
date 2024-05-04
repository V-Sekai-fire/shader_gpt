using UnityEngine;

namespace ShaderGPT {
[System.Serializable]
public class GenerationConfig {
	public int max_length = 2048;
	public bool do_sample = false;
	public float temperature = 1f;
	public float repetition_penalty = 1f;
}
public abstract class ModelForCausalLM<T> : PretrainedModel<T>, ModelForCausalLM where T : PretrainedConfig<T> {
	public GenerationConfig generation_config {get;set;}
	public ModelForCausalLM(TensorNN nn, T config): base(nn, config) {
		generation_config = new GenerationConfig(); // TODO: parse config
	}
	public abstract (Texture, Texture) ForCausalLM(Texture input_ids);

	protected int max_length => generation_config.max_length;
	private float temperature => generation_config.do_sample ? generation_config.temperature : 0f;
	private float repetition_penalty => generation_config.repetition_penalty;
	void RepetitionPenaltyLogitsProcessor(Texture input_ids, ref Texture scores, float penalty, Texture last_input_ids) {
		var inputs_T = nn.Transpose(input_ids, 1);
		inputs_T = BatchRelease(nn.Fusion(MarkRelease(inputs_T), @default:4*ctx.Size1(scores)*Vector4.one,
			window:(new Vector4(-max_length, ctx.Size0(last_input_ids), 0, 1), last_input_ids)));
		var mask = nn.Fusion(scores, scale:0f);
		BatchRelease(nn.IndexCopy((RenderTexture)mask, (MarkRelease(inputs_T), 0), null, fill:1f, axis1:true));
		var penal = nn.Fusion(scores, func:TensorNN.Keyword.FUNC_RELU, eps:penalty*penalty, scale:1f/penalty);
		var diff = BatchRelease(nn.Fusion(scores, scale:-1, add:MarkRelease(penal)));
		scores = BatchRelease(nn.Fusion(MarkRelease(mask), mul:MarkRelease(diff), add:MarkRelease(scores)));
	}
	public Texture Generate(Texture input, ref Texture scores) {
		var inputs = ctx.PersistentGPUTensor("inputs", max_length, 1);
		nn.IndexCopy(inputs, (input, 1), input);
		if(ctx.Size0(scores) > 1)
			scores = BatchRelease(nn.Fusion(ctx.Slice(MarkRelease(scores), 1, ctx.Size1(scores), ctx.Size0(scores)-1, 0)));
		RepetitionPenaltyLogitsProcessor(inputs, ref scores, repetition_penalty, input);
		var gumbel = BatchRelease(nn.Gumbel(MarkRelease(scores), temperature));
		return BatchRelease(nn.ArgMax(MarkRelease(gumbel), window:(new Vector2(0, config.vocab_size), null)));
	}
}
public interface ModelForCausalLM : PretrainedModel {
	public GenerationConfig generation_config {get;set;}
	public (Texture, Texture) ForCausalLM(Texture input_ids);
	public Texture Generate(Texture input, ref Texture scores);

	public static ModelForCausalLM FromPretrained(TensorNN nn, TextAsset configJson) {
		var config = JsonUtility.FromJson<PretrainedConfig>(configJson.text);
		switch(config.model_type) {
		case "gpt2":
			return new Models.GPT2(nn, Models.GPT2Config.FromPretrained(configJson));
		case "gpt_neo":
			return new Models.GPTNeo(nn, Models.GPTNeoConfig.FromPretrained(configJson));
		case "gpt_neox":
			return new Models.GPTNeoX(nn, Models.GPTNeoXConfig.FromPretrained(configJson));
		case "gemma":
		case "llama":
		case "mistral":
		case "qwen2":
			return new Models.Llama(nn, Models.LlamaConfig.FromPretrained(configJson));
		case "openelm":
			return new Models.OpenELM(nn, Models.OpenELMConfig.FromPretrained(configJson));
		case "phi":
			return new Models.Phi(nn, Models.PhiConfig.FromPretrained(configJson));
		case "phi3":
			return new Models.Phi3(nn, Models.Phi3Config.FromPretrained(configJson));
		default:
			throw new System.NotSupportedException($"unsupported architecture \"{config.model_type}\"");
		}
	}
	public static ModelForCausalLM FromPretrained(TensorNN nn, TextAsset configJson, Texture[] textures) {
		var model = FromPretrained(nn, configJson);
		model.LoadStateDict(textures);
		return model;
	}
}
}