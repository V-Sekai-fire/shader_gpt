using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class GPTNeoXConfig {
	public int num_hidden_layers;
	public int num_attention_heads;
	public float layer_norm_eps;
	public string hidden_act;
	public int max_position_embeddings;
	public bool use_parallel_residual;
}
public class GPTNeoX : ModelForCausalLM {
	public GPTNeoXConfig config;
	public GPTNeoX(TensorNN nn, TextAsset configJson): base(nn, configJson) {
		config = JsonUtility.FromJson<GPTNeoXConfig>(configJson.text);
		maxLength = Mathf.Min(maxLength, config.max_position_embeddings);
	}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => GPTNeoXForCausalLM(input_ids);

	void GPTNeoXAttention(ref Texture hidden_states, Texture input_ids, string path) {
		var query = nn.Linear(hidden_states, state_dict[$"{path}.query.weight"], state_dict[$"{path}.query.bias"]);
		var key   = nn.Linear(hidden_states, state_dict[$"{path}.key.weight"],   state_dict[$"{path}.key.bias"]);
		var value = nn.Linear(hidden_states, state_dict[$"{path}.value.weight"], state_dict[$"{path}.value.bias"]);
		ctx.Release(hidden_states);

		state_dict.TryGetValue(Regex.Replace($"{path}.rotary_emb.weight", @"[.]\d+[.]", ".0."), out var rotary_emb);
		var rotary = nn.IndexSelect(rotary_emb ?? state_dict[$"{path}.rotary_emb.weight"], (input_ids, 1));
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:config.num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:config.num_attention_heads));
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), MarkRelease(value)));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(ctx.Size1(query)*4 / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:new Vector4(1-window_size, 1, 0, 1), offset:input_ids));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightT:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.dense.weight"], state_dict[$"{path}.dense.bias"]));
	}
	void GPTNeoXMLP(ref Texture hidden_states, string path) {
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.dense_h_to_4h.weight"], state_dict[$"{path}.dense_h_to_4h.bias"]));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.ActFn(config.hidden_act)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.dense_4h_to_h.weight"], state_dict[$"{path}.dense_4h_to_h.bias"]));
	}
	void GPTNeoXLayer(ref Texture hidden_states, Texture input_ids, string path) {
		Debug.Assert(config.use_parallel_residual, "only use_parallel_residual=true is implemented");
		var attn_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.input_layernorm.weight"], state_dict[$"{path}.input_layernorm.bias"], config.layer_norm_eps);
		GPTNeoXAttention(ref attn_states, input_ids, path:$"{path}.attention");
		var mlp_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.post_attention_layernorm.weight"], state_dict[$"{path}.post_attention_layernorm.bias"], config.layer_norm_eps);
		GPTNeoXMLP(ref mlp_states, path:$"{path}.mlp");
		var sum = BatchRelease(nn.Fusion(MarkRelease(attn_states), add:MarkRelease(mlp_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(sum)));
	}
	Texture GPTNeoXModel(Texture input_ids, string path) {
		var hidden_states = nn.IndexSelect(state_dict[$"{path}.embed_in.weight.T"], (input_ids, 0), inputT:true);
		for(int i=0; i<config.num_hidden_layers; i++)
			GPTNeoXLayer(ref hidden_states, input_ids, path:$"{path}.layers.{i}");
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), state_dict[$"{path}.final_layer_norm.weight"], state_dict[$"{path}.final_layer_norm.bias"], config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) GPTNeoXForCausalLM(Texture input_ids) {
		var hidden_states = GPTNeoXModel(input_ids, path:"gpt_neox");
		var lm_logits = nn.Linear(hidden_states, state_dict["embed_out.weight.T"], weightT:true);
		return (hidden_states, lm_logits);
	}
}
}