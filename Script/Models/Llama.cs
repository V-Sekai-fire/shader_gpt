using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class LlamaConfig : PretrainedConfig<LlamaConfig>, ISerializationCallbackReceiver {
	public int max_position_embeddings;
	public int hidden_size;
	public int num_hidden_layers;
	public int num_attention_heads;
	public int num_key_value_heads;
	public string hidden_act;
	public float rms_norm_eps;
	public int sliding_window; // for mistral & qwen2

	public string hidden_activation; // for gemma
	public void OnBeforeSerialize() {}
	public void OnAfterDeserialize() {
		if(model_type == "gemma")
			hidden_act = string.IsNullOrEmpty(hidden_activation) ? "gelu_pytorch_tanh" : hidden_activation;
	}
}
public class Llama : ModelForCausalLM<LlamaConfig> {
	public Llama(TensorNN nn, LlamaConfig config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => LlamaForCausalLM(input_ids);

	void LlamaAttention(string path, ref Texture hidden_states, Texture input_ids) {
		state_dict.TryGetValue($"{path}.q_proj.bias", out var q_bias);
		state_dict.TryGetValue($"{path}.k_proj.bias", out var k_bias);
		state_dict.TryGetValue($"{path}.v_proj.bias", out var v_bias);
		var query = nn.Linear(hidden_states, state_dict[$"{path}.q_proj.weight"], q_bias);
		var key   = nn.Linear(hidden_states, state_dict[$"{path}.k_proj.weight"], k_bias);
		var value = nn.Linear(hidden_states, state_dict[$"{path}.v_proj.weight"], v_bias);
		ctx.Release(hidden_states);

		state_dict.TryGetValue(Regex.Replace($"{path}.rotary_emb.weight", @"[.]\d+[.]", ".0."), out var rotary_emb);
		var rotary = nn.IndexSelect(rotary_emb ?? state_dict[$"{path}.rotary_emb.weight"], (input_ids, 1));
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:config.num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:config.num_key_value_heads));
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", max_length, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", max_length, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), MarkRelease(value)));

		var window_size = config.sliding_window == 0 ? config.max_position_embeddings : config.sliding_window;
		var norm_factor = 1f / Mathf.Sqrt(config.hidden_size / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 0, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads, weightT:true));
		state_dict.TryGetValue($"{path}.o_proj.bias", out var o_bias);
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.o_proj.weight"], o_bias));
	}
	void LlamaMLP(string path, ref Texture hidden_states) {
		var gate = nn.Linear(hidden_states, state_dict[$"{path}.gate_proj.weight"]);
		var up   = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.up_proj.weight"]));
		var act  = BatchRelease(nn.Fusion(MarkRelease(gate), func:TensorNN.ActFn(config.hidden_act)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(up), mul:MarkRelease(act)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.down_proj.weight"]));
	}
	void LlamaDecoderLayer(string path, ref Texture hidden_states, Texture input_ids, float scale=1f) {
		var attn_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.input_layernorm.weight"], null, config.rms_norm_eps, rmsNorm:true);
		LlamaAttention($"{path}.self_attn", ref attn_states, input_ids);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), scale:scale, add:MarkRelease(attn_states)));
		var mlp_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.post_attention_layernorm.weight"], null, config.rms_norm_eps, rmsNorm:true);
		LlamaMLP($"{path}.mlp", ref mlp_states);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture LlamaModel(string path, Texture input_ids) {
		FixSize0($"{path}.embed_tokens.weight.T", config.hidden_size);
		var hidden_states = nn.IndexSelect(state_dict[$"{path}.embed_tokens.weight.T"], (input_ids, 0), inputT:true);
		for(int i=0; i<config.num_hidden_layers; i++)
			LlamaDecoderLayer($"{path}.layers.{i}", ref hidden_states, input_ids,
				scale:(i == 0 && config.model_type == "gemma" ? Mathf.Sqrt(config.hidden_size) : 1f));
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), state_dict[$"{path}.norm.weight"], null, config.rms_norm_eps, rmsNorm:true));
		return hidden_states;
	}
	(Texture, Texture) LlamaForCausalLM(Texture input_ids) {
		FixSize0("lm_head.weight.T", config.hidden_size);
		var hidden_states = LlamaModel("model", input_ids);
		var logits = nn.Linear(hidden_states, state_dict["lm_head.weight.T"], weightT:true);
		return (hidden_states, logits);
	}
}
}