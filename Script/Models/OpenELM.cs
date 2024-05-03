using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class OpenELMConfig : PretrainedConfig {
	public int max_context_length;
	public int model_dim;
	public int num_transformer_layers;
	public int[] num_query_heads;
	public int[] num_kv_heads;
	public string activation_fn_name;
	public int head_dim;

	// TODO: clean up when openelm is added to transformers
	public int max_position_embeddings => max_context_length;
	public int hidden_size => model_dim;
	public int num_hidden_layers => num_transformer_layers;
	public string hidden_act => activation_fn_name;
	public float rms_norm_eps => 1e-6f; // OpenELMRMSNorm default
}
public class OpenELM : ModelForCausalLM<OpenELMConfig> {
	public OpenELM(TensorNN nn, TextAsset configJson): base(nn, configJson) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => OpenELMForCausalLM(input_ids);

	void OpenELMMultiHeadCausalAttention(ref Texture hidden_states, Texture input_ids, string path, int layer_id) {
		var head_dim = config.head_dim;
		var num_attention_heads = config.num_query_heads[layer_id];
		var num_key_value_heads = config.num_kv_heads[layer_id];

		var qkv = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.qkv_proj.weight"]));
		var q = ctx.Slice(qkv, ctx.Size0(qkv), num_attention_heads*head_dim/4);
		var k = ctx.Slice(qkv, ctx.Size0(qkv), num_key_value_heads*head_dim/4, 0, num_attention_heads*head_dim/4);
		var v = ctx.Slice(qkv, ctx.Size0(qkv), num_key_value_heads*head_dim/4, 0, (num_attention_heads+num_key_value_heads)*head_dim/4);
		var query = nn.GroupNorm(q, state_dict[$"{path}.q_norm.weight"], null, config.rms_norm_eps, rmsNorm:true, groups:num_attention_heads);
		var key   = nn.GroupNorm(k, state_dict[$"{path}.k_norm.weight"], null, config.rms_norm_eps, rmsNorm:true, groups:num_key_value_heads);
		var value = nn.Fusion(v);
		ctx.Release(qkv);

		state_dict.TryGetValue(Regex.Replace($"{path}.pos_embedding.weight", @"[.]\d+[.]", ".0."), out var rotary_emb);
		var rotary = nn.IndexSelect(rotary_emb ?? state_dict[$"{path}.pos_embedding.weight"], (input_ids, 1));
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:num_key_value_heads));
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", max_length, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", max_length, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), MarkRelease(value)));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(head_dim);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:num_attention_heads, weightHeads:num_key_value_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:num_attention_heads, window:(new Vector4(1-window_size, 1, 0, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:num_attention_heads, weightHeads:num_key_value_heads, weightT:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.out_proj.weight"]));
	}
	void OpenELMFeedForwardNetwork(ref Texture hidden_states, string path) {
		var y_12 = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.proj_1.weight"]));
		var y_1 = ctx.Slice(y_12, ctx.Size0(y_12), ctx.Size1(y_12)/2);
		var y_2 = ctx.Slice(y_12, ctx.Size0(y_12), ctx.Size1(y_12)/2, 0, ctx.Size1(y_12)/2);
		var act = nn.Fusion(y_1, func:TensorNN.ActFn(config.hidden_act));
		hidden_states = BatchRelease(nn.Fusion((MarkRelease(y_12), y_2).Item2, mul:MarkRelease(act)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.proj_2.weight"]));
	}
	void OpenELMDecoderLayer(ref Texture hidden_states, Texture input_ids, string path, int layer_id) {
		var attn_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.attn_norm.weight"], null, config.rms_norm_eps, rmsNorm:true);
		OpenELMMultiHeadCausalAttention(ref attn_states, input_ids, path:$"{path}.attn", layer_id:layer_id);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		var mlp_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.ffn_norm.weight"], null, config.rms_norm_eps, rmsNorm:true);
		OpenELMFeedForwardNetwork(ref mlp_states, path:$"{path}.ffn");
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture OpenELMModel(Texture input_ids, string path) {
		var hidden_states = nn.IndexSelect(state_dict[$"{path}.token_embeddings.weight.T"], (input_ids, 0), inputT:true);
		for(int i=0; i<config.num_hidden_layers; i++)
			OpenELMDecoderLayer(ref hidden_states, input_ids, path:$"{path}.layers.{i}", layer_id:i);
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), state_dict[$"{path}.norm.weight"], null, config.rms_norm_eps, rmsNorm:true));
		return hidden_states;
	}
	(Texture, Texture) OpenELMForCausalLM(Texture input_ids) {
		var hidden_states = OpenELMModel(input_ids, path:"transformer");
		state_dict.TryGetValue("lm_head.weight.T", out var lm_head);
		var logits = nn.Linear(hidden_states, lm_head ?? state_dict["transformer.token_embeddings.weight.T"], weightT:true);
		return (hidden_states, logits);
	}
}
}