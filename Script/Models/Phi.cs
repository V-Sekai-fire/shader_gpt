using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class PhiConfig : PretrainedConfig<PhiConfig> {
	public int max_position_embeddings;
	public int hidden_size;
	public int num_hidden_layers;
	public int num_attention_heads;
	public int num_key_value_heads;
	public string hidden_act;
	public float layer_norm_eps;
}
public class Phi : ModelForCausalLM<PhiConfig> {
	public Phi(TensorNN nn, PhiConfig config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => PhiForCausalLM(input_ids);

	void PhiAttention(string path, ref Texture hidden_states, Texture input_ids) {
		var query = Linear($"{path}.q_proj", hidden_states);
		var key   = Linear($"{path}.k_proj", hidden_states);
		var value = Linear($"{path}.v_proj", hidden_states);
		ctx.Release(hidden_states);

		var rotary = Embedding($"{path}.rotary_emb", (input_ids, 1), fallback:Regex.Replace($"{path}.rotary_emb", @"[.]\d+[.]", ".0."));
		query = BatchRelease(nn.Rotary(MarkRelease(query), rotary, groups:config.num_attention_heads));
		key   = BatchRelease(nn.Rotary(MarkRelease(key),   rotary, groups:config.num_key_value_heads));
		ctx.Release(rotary);
		var keys   = BatchRelease(CacheUpdate($"{path}.k", (input_ids, 1), MarkRelease(key)));
		var values = BatchRelease(CacheUpdate($"{path}.v", (input_ids, 1), MarkRelease(value)));

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(config.hidden_size / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 0, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads, weightT:true));
		hidden_states = BatchRelease(Linear($"{path}.dense", MarkRelease(hidden_states)));
	}
	Texture PhiMLP(string path, Texture hidden_states) {
		hidden_states = Linear($"{path}.fc1", hidden_states);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.ActFn(config.hidden_act)));
		hidden_states = BatchRelease(Linear($"{path}.fc2", MarkRelease(hidden_states)));
		return hidden_states;
	}
	void PhiDecoderLayer(string path, ref Texture hidden_states, Texture input_ids) {
		var attn_states = LayerNorm($"{path}.input_layernorm", hidden_states, config.layer_norm_eps);
		var mlp_states = PhiMLP($"{path}.mlp", attn_states);
		PhiAttention($"{path}.self_attn", ref attn_states, input_ids);
		var sum = BatchRelease(nn.Fusion(MarkRelease(attn_states), add:MarkRelease(mlp_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(sum)));
	}
	Texture PhiModel(string path, Texture input_ids) {
		var hidden_states = Embedding($"{path}.embed_tokens", (input_ids, 0));
		for(int i=0; i<config.num_hidden_layers; i++)
			PhiDecoderLayer($"{path}.layers.{i}", ref hidden_states, input_ids);
		hidden_states = BatchRelease(LayerNorm($"{path}.final_layernorm", MarkRelease(hidden_states), config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) PhiForCausalLM(Texture input_ids) {
		var hidden_states = PhiModel("model", input_ids);
		var logits = Linear("lm_head", hidden_states);
		return (logits, hidden_states);
	}
}
}