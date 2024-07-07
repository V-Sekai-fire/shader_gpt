using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;

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
	public Dictionary<string,RenderTexture> cache {get;set;}
	public ModelForCausalLM(TensorNN nn, T config): base(nn, config) {
		generation_config = new GenerationConfig(); // TODO: parse config
		cache = new Dictionary<string,RenderTexture>();
	}
	public abstract (Texture logits, Texture hidden_states) ForCausalLM(Texture input_ids);

	private int max_length => generation_config.max_length;
	private float temperature => generation_config.do_sample ? generation_config.temperature : 0f;
	private float repetition_penalty => generation_config.repetition_penalty;
	void RepetitionPenaltyLogitsProcessor(Texture input_ids, ref Texture scores, float penalty, Texture last_input_ids) {
		var inputs_T = nn.Transpose(input_ids, 1);
		inputs_T = BatchRelease(nn.Fusion(MarkRelease(inputs_T), @default:4*ctx.Size1(scores)*Vector4.one,
			window:(new Vector4(-max_length, ctx.Size0(last_input_ids), 1, 1), last_input_ids)));
		var mask = nn.Fusion(scores, scale:0f);
		BatchRelease(nn.IndexCopy((RenderTexture)mask, (MarkRelease(inputs_T), 0), null, 1f, axis1:true));
		var penal = nn.Fusion(scores, func:TensorNN.Keyword.FUNC_RELU, eps:penalty*penalty, scale:1f/penalty);
		var diff = BatchRelease(nn.Fusion(scores, scale:-1, add:MarkRelease(penal)));
		scores = BatchRelease(nn.Fusion(MarkRelease(mask), mul:MarkRelease(diff), add:MarkRelease(scores)));
	}
	public Texture Generate(Texture input, ref Texture scores) {
		var inputs = CacheUpdate("inputs", (input, 1), input);
		scores = BatchRelease(nn.Fusion(ctx.Slice(MarkRelease(scores), 1, ctx.Size1(scores), ctx.Size0(scores)-1, 0),
			dtype:VertexAttributeFormat.Float32)); // cast to float32
		RepetitionPenaltyLogitsProcessor(inputs, ref scores, repetition_penalty, input);
		var gumbel = BatchRelease(nn.Gumbel(MarkRelease(scores), temperature));
		return BatchRelease(nn.ArgMax(MarkRelease(gumbel), window:(new Vector2(0, config.vocab_size), default)));
	}

	protected Texture CacheUpdate(string name, (TexView, int) position, TexView state) {
		if(!cache.TryGetValue(name, out var states))
			cache[name] = states = ctx.GPUTensor(max_length, ctx.Size1(state), dtype:ctx.DType(state));
		nn.IndexCopy(states, position, state);
		return states;
	}
	protected Texture CacheUpdate(string name, TexView state) {
		if(cache.TryGetValue(name, out var states))
			ctx.Release(states);
		cache[name] = states = ctx.GPUTensor(ctx.Size0(state), ctx.Size1(state), dtype:ctx.DType(state));
		return (Texture)nn.Copy(states, state);
	}
	public void CacheClear() {
		foreach(var x in cache)
			ctx.Release(x.Value);
		cache.Clear();
	}
}
public interface ModelForCausalLM : PretrainedModel {
	/*public*/ GenerationConfig generation_config {get;set;}
	/*public*/ Dictionary<string,RenderTexture> cache {get;set;}
	/*public*/ (Texture logits, Texture hidden_states) ForCausalLM(Texture input_ids);
	/*public*/ Texture Generate(Texture input, ref Texture scores);
	/*public*/ void CacheClear();
}
public abstract class ModelForSeq2SeqLM<T> : ModelForCausalLM<T>, ModelForSeq2SeqLM where T : PretrainedConfig<T> {
	public ModelForSeq2SeqLM(TensorNN nn, T config): base(nn, config) {}
	public abstract (Texture logits, Texture decoder_hidden_states, Texture encoder_hidden_states) ForSeq2SeqLM(Texture input_ids, Texture decoder_input_ids, Texture encoder_hidden_states=null);
	public override (Texture logits, Texture hidden_states) ForCausalLM(Texture input_ids) {
		var o = ForSeq2SeqLM(null, input_ids);
		return (o.logits, o.decoder_hidden_states);
	}
}
public interface ModelForSeq2SeqLM : ModelForCausalLM {
	/*public*/ (Texture logits, Texture decoder_hidden_states, Texture encoder_hidden_states) ForSeq2SeqLM(Texture input_ids, Texture decoder_input_ids, Texture encoder_hidden_states=null);
}
public static class AutoModelForCausalLM {
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
		case "stablelm":
		case "phi3":
			return new Models.Llama(nn, Models.LlamaConfig.FromPretrained(configJson));
		case "openelm":
			return new Models.OpenELM(nn, Models.OpenELMConfig.FromPretrained(configJson));
		case "phi":
			return new Models.Phi(nn, Models.PhiConfig.FromPretrained(configJson));
		case "t5":
			return new Models.T5(nn, Models.T5Config.FromPretrained(configJson));
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