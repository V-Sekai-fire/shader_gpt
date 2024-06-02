using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class Phi3Config : PretrainedConfig<Phi3Config> {
	public int max_position_embeddings;
	public int hidden_size;
	public int num_hidden_layers;
	public int num_attention_heads;
	public int num_key_value_heads;
	public string hidden_act;
	public float rms_norm_eps;
	public int head_dim;
}
public class Phi3 : ModelForCausalLM<Phi3Config> {
	public Phi3(TensorNN nn, Phi3Config config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => Phi3ForCausalLM(input_ids);

	void Phi3Attention(string path, ref Texture hidden_states, Texture input_ids) {
		var head_dim = config.hidden_size / config.num_attention_heads;
		var qkv = BatchRelease(Linear($"{path}.qkv_proj", MarkRelease(hidden_states)));
		var q = ctx.Slice(qkv, ctx.Size0(qkv), config.num_attention_heads*head_dim/4);
		var k = ctx.Slice(qkv, ctx.Size0(qkv), config.num_key_value_heads*head_dim/4, 0, config.num_attention_heads*head_dim/4);
		var v = ctx.Slice(qkv, ctx.Size0(qkv), config.num_key_value_heads*head_dim/4, 0, (config.num_attention_heads+config.num_key_value_heads)*head_dim/4);

		var rotary = Embedding($"{path}.rotary_emb", (input_ids, 1), fallback:Regex.Replace($"{path}.rotary_emb", @"[.]\d+[.]", ".0."));
		var query = nn.Rotary(q, rotary, groups:config.num_attention_heads);
		var key   = nn.Rotary(k, rotary, groups:config.num_key_value_heads);
		ctx.Release(rotary);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", max_length, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", max_length, ctx.Size1(v), dtype:ctx.DType(v));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), (MarkRelease(qkv), v).Item2));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(head_dim);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 0, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads, weightT:true));
		hidden_states = BatchRelease(Linear($"{path}.o_proj", MarkRelease(hidden_states)));
	}
	void Phi3MLP(string path, ref Texture hidden_states) {
		var gate_up = BatchRelease(Linear($"{path}.gate_up_proj", MarkRelease(hidden_states)));
		var gate = ctx.Slice(gate_up, ctx.Size0(gate_up), ctx.Size1(gate_up)/2);
		var up   = ctx.Slice(gate_up, ctx.Size0(gate_up), ctx.Size1(gate_up)/2, 0, ctx.Size1(gate_up)/2);
		var act  = nn.Fusion(gate, func:TensorNN.ActFn(config.hidden_act));
		hidden_states = BatchRelease(nn.Fusion((MarkRelease(gate_up), up).Item2, mul:MarkRelease(act)));
		hidden_states = BatchRelease(Linear($"{path}.down_proj", MarkRelease(hidden_states)));
	}
	void Phi3DecoderLayer(string path, ref Texture hidden_states, Texture input_ids) {
		var attn_states = LayerNorm($"{path}.input_layernorm", hidden_states, config.rms_norm_eps, rms:true);
		Phi3Attention($"{path}.self_attn", ref attn_states, input_ids);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		var mlp_states = LayerNorm($"{path}.post_attention_layernorm", hidden_states, config.rms_norm_eps, rms:true);
		Phi3MLP($"{path}.mlp", ref mlp_states);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture Phi3Model(string path, Texture input_ids) {
		var hidden_states = Embedding($"{path}.embed_tokens", (input_ids, 0));
		for(int i=0; i<config.num_hidden_layers; i++)
			Phi3DecoderLayer($"{path}.layers.{i}", ref hidden_states, input_ids);
		hidden_states = BatchRelease(LayerNorm($"{path}.norm", MarkRelease(hidden_states), config.rms_norm_eps, rms:true));
		return hidden_states;
	}
	(Texture, Texture) Phi3ForCausalLM(Texture input_ids) {
		var hidden_states = Phi3Model("model", input_ids);
		var logits = Linear("lm_head", hidden_states);
		return (hidden_states, logits);
	}
}
}