using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class OpenELMConfig : PretrainedConfig<OpenELMConfig> {
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
	public OpenELM(TensorNN nn, OpenELMConfig config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => OpenELMForCausalLM(input_ids);

	void OpenELMMultiHeadCausalAttention(string path, ref Texture hidden_states, Texture input_ids, int layer_id) {
		var head_dim = config.head_dim;
		var num_attention_heads = config.num_query_heads[layer_id];
		var num_key_value_heads = config.num_kv_heads[layer_id];

		var qkv = BatchRelease(Linear($"{path}.qkv_proj", MarkRelease(hidden_states)));
		var q = ctx.Slice(qkv, ctx.Size0(qkv), num_attention_heads*head_dim/4);
		var k = ctx.Slice(qkv, ctx.Size0(qkv), num_key_value_heads*head_dim/4, 0, num_attention_heads*head_dim/4);
		var v = ctx.Slice(qkv, ctx.Size0(qkv), num_key_value_heads*head_dim/4, 0, (num_attention_heads+num_key_value_heads)*head_dim/4);
		var query = LayerNorm($"{path}.q_norm", q, config.rms_norm_eps, rms:true, groups:num_attention_heads);
		var key   = LayerNorm($"{path}.k_norm", k, config.rms_norm_eps, rms:true, groups:num_key_value_heads);
		var value = nn.Fusion(v);
		ctx.Release(qkv);

		var rotary = Embedding($"{path}.pos_embedding", (input_ids, 1), fallback:Regex.Replace($"{path}.pos_embedding", @"[.]\d+[.]", ".0."));
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
		hidden_states = BatchRelease(Linear($"{path}.o_proj", MarkRelease(hidden_states)));
	}
	void OpenELMFeedForwardNetwork(string path, ref Texture hidden_states) {
		var gate_up = BatchRelease(Linear($"{path}.proj_1", MarkRelease(hidden_states)));
		var gate = ctx.Slice(gate_up, ctx.Size0(gate_up), ctx.Size1(gate_up)/2);
		var up   = ctx.Slice(gate_up, ctx.Size0(gate_up), ctx.Size1(gate_up)/2, 0, ctx.Size1(gate_up)/2);
		var act  = nn.Fusion(gate, func:TensorNN.ActFn(config.hidden_act));
		hidden_states = BatchRelease(nn.Fusion((MarkRelease(gate_up), up).Item2, mul:MarkRelease(act)));
		hidden_states = BatchRelease(Linear($"{path}.proj_2", MarkRelease(hidden_states)));
	}
	void OpenELMDecoderLayer(string path, ref Texture hidden_states, Texture input_ids, int layer_id) {
		var attn_states = LayerNorm($"{path}.attn_norm", hidden_states, config.rms_norm_eps, rms:true);
		OpenELMMultiHeadCausalAttention($"{path}.attn", ref attn_states, input_ids, layer_id:layer_id);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		var mlp_states = LayerNorm($"{path}.ffn_norm", hidden_states, config.rms_norm_eps, rms:true);
		OpenELMFeedForwardNetwork($"{path}.ffn", ref mlp_states);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture OpenELMModel(string path, Texture input_ids) {
		var hidden_states = Embedding($"{path}.token_embeddings", (input_ids, 0));
		for(int i=0; i<config.num_hidden_layers; i++)
			OpenELMDecoderLayer($"{path}.layers.{i}", ref hidden_states, input_ids, layer_id:i);
		hidden_states = BatchRelease(LayerNorm($"{path}.norm", MarkRelease(hidden_states), config.rms_norm_eps, rms:true));
		return hidden_states;
	}
	(Texture, Texture) OpenELMForCausalLM(Texture input_ids) {
		var hidden_states = OpenELMModel("transformer", input_ids);
		var logits = Linear("lm_head", hidden_states, fallback:"transformer.token_embeddings");
		return (hidden_states, logits);
	}
}
}