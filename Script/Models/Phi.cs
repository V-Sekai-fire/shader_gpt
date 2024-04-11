using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class PhiConfig {
	public int num_hidden_layers;
	public int num_attention_heads;
	public int num_key_value_heads;
	public float layer_norm_eps;
	public string hidden_act;
	public int max_position_embeddings;
}
public class Phi : ModelForCausalLM {
	public PhiConfig config;
	public Phi(TensorNN nn, TextAsset configJson): base(nn, configJson) {
		config = JsonUtility.FromJson<PhiConfig>(configJson.text);
		maxLength = Mathf.Min(maxLength, config.max_position_embeddings);
	}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => PhiForCausalLM(input_ids);

	void PhiAttention(ref Texture hidden_states, Texture input_ids, string path) {
		var query = nn.Linear(hidden_states, state_dict[$"{path}.q_proj.weight"], state_dict[$"{path}.q_proj.bias"]);
		var key   = nn.Linear(hidden_states, state_dict[$"{path}.k_proj.weight"], state_dict[$"{path}.k_proj.bias"]);
		var value = nn.Linear(hidden_states, state_dict[$"{path}.v_proj.weight"], state_dict[$"{path}.v_proj.bias"]);
		ctx.Release(hidden_states);

		state_dict.TryGetValue(Regex.Replace($"{path}.rotary_emb.weight", @"[.]\d+[.]", ".0."), out var rotary_emb);
		var rotary = nn.IndexSelect(rotary_emb ?? state_dict[$"{path}.rotary_emb.weight"], (input_ids, 1));
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:config.num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:config.num_key_value_heads));
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), MarkRelease(value)));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(ctx.Size1(query)*4 / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:new Vector4(1-window_size, 1, 0, 1), offset:input_ids));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads, weightT:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.dense.weight"], state_dict[$"{path}.dense.bias"]));
	}
	void PhiDecoderLayer(ref Texture hidden_states, Texture input_ids, string path) {
		var attn_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.input_layernorm.weight"], state_dict[$"{path}.input_layernorm.bias"], config.layer_norm_eps);
		var mlp_states = BatchRelease(nn.Linear(attn_states, state_dict[$"{path}.mlp.fc1.weight"], state_dict[$"{path}.mlp.fc1.bias"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), func:TensorNN.ActFn(config.hidden_act)));
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), state_dict[$"{path}.mlp.fc2.weight"], state_dict[$"{path}.mlp.fc2.bias"]));

		PhiAttention(ref attn_states, input_ids, path:$"{path}.self_attn");
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture PhiModel(Texture input_ids, string path) {
		var hidden_states = nn.IndexSelect(state_dict[$"{path}.embed_tokens.weight.T"], (input_ids, 0), inputT:true);
		for(int i=0; i<config.num_hidden_layers; i++)
			PhiDecoderLayer(ref hidden_states, input_ids, path:$"{path}.layers.{i}");
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), state_dict[$"{path}.final_layernorm.weight"], state_dict[$"{path}.final_layernorm.bias"], config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) PhiForCausalLM(Texture input_ids) {
		var hidden_states = PhiModel(input_ids, path:"model");
		var lm_logits = nn.Linear(hidden_states, state_dict["lm_head.weight.T"], state_dict["lm_head.bias"], weightT:true);
		return (hidden_states, lm_logits);
	}
}
}