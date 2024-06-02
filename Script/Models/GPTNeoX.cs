using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class GPTNeoXConfig : PretrainedConfig<GPTNeoXConfig> {
	public int max_position_embeddings;
	public int hidden_size;
	public int num_hidden_layers;
	public int num_attention_heads;
	public string hidden_act;
	public float layer_norm_eps;
	public bool use_parallel_residual;
}
public class GPTNeoX : ModelForCausalLM<GPTNeoXConfig> {
	public GPTNeoX(TensorNN nn, GPTNeoXConfig config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => GPTNeoXForCausalLM(input_ids);

	void GPTNeoXAttention(string path, ref Texture hidden_states, Texture input_ids) {
		var qkv = BatchRelease(Linear($"{path}.query_key_value", MarkRelease(hidden_states)));
		var q = ctx.Slice(qkv, ctx.Size0(qkv), config.hidden_size/4);
		var k = ctx.Slice(qkv, ctx.Size0(qkv), config.hidden_size/4, 0, config.hidden_size/4);
		var v = ctx.Slice(qkv, ctx.Size0(qkv), config.hidden_size/4, 0, config.hidden_size/2);

		var rotary = Embedding($"{path}.rotary_emb", (input_ids, 1), fallback:Regex.Replace($"{path}.rotary_emb", @"[.]\d+[.]", ".0."));
		var query = nn.Rotary(q, rotary, groups:config.num_attention_heads);
		var key   = nn.Rotary(k, rotary, groups:config.num_attention_heads);
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", max_length, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", max_length, ctx.Size1(v), dtype:ctx.DType(v));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), (MarkRelease(qkv), v).Item2));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(config.hidden_size / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 0, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightT:true));
		hidden_states = BatchRelease(Linear($"{path}.dense", MarkRelease(hidden_states)));
	}
	void GPTNeoXMLP(string path, ref Texture hidden_states) {
		hidden_states = BatchRelease(Linear($"{path}.dense_h_to_4h", MarkRelease(hidden_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.ActFn(config.hidden_act)));
		hidden_states = BatchRelease(Linear($"{path}.dense_4h_to_h", MarkRelease(hidden_states)));
	}
	void GPTNeoXLayer(string path, ref Texture hidden_states, Texture input_ids) {
		Debug.Assert(config.use_parallel_residual, "only use_parallel_residual=true is implemented");
		var attn_states = LayerNorm($"{path}.input_layernorm", hidden_states, config.layer_norm_eps);
		GPTNeoXAttention($"{path}.attention", ref attn_states, input_ids);
		var mlp_states = LayerNorm($"{path}.post_attention_layernorm", hidden_states, config.layer_norm_eps);
		GPTNeoXMLP($"{path}.mlp", ref mlp_states);
		var sum = BatchRelease(nn.Fusion(MarkRelease(attn_states), add:MarkRelease(mlp_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(sum)));
	}
	Texture GPTNeoXModel(string path, Texture input_ids) {
		var hidden_states = Embedding($"{path}.embed_in", (input_ids, 0));
		for(int i=0; i<config.num_hidden_layers; i++)
			GPTNeoXLayer($"{path}.layers.{i}", ref hidden_states, input_ids);
		hidden_states = BatchRelease(LayerNorm($"{path}.final_layer_norm", MarkRelease(hidden_states), config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) GPTNeoXForCausalLM(Texture input_ids) {
		var hidden_states = GPTNeoXModel("gpt_neox", input_ids);
		var logits = Linear("embed_out", hidden_states);
		return (hidden_states, logits);
	}
}
}