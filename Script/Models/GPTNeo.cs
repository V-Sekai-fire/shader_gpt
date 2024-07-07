using UnityEngine;

namespace ShaderGPT.Models {
[System.Serializable]
public class GPTNeoConfig : PretrainedConfig<GPTNeoConfig> {
	public int max_position_embeddings;
	public int hidden_size;
	public int num_layers;
	public int num_heads;
	public int window_size;
	public string activation_function;
	public float layer_norm_epsilon;
	public string[] attention_layers;

	public int num_attention_heads => num_heads;
	public int num_hidden_layers => num_layers;
	public string hidden_act => activation_function;
	public float layer_norm_eps => layer_norm_epsilon;
}
public class GPTNeo : ModelForCausalLM<GPTNeoConfig> {
	public GPTNeo(TensorNN nn, GPTNeoConfig config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => GPTNeoForCausalLM(input_ids);

	void GPTNeoSelfAttention(string path, ref Texture hidden_states, Texture input_ids, int layer_id) {
		var query = Linear($"{path}.q_proj", hidden_states);
		var key   = Linear($"{path}.k_proj", hidden_states);
		var value = Linear($"{path}.v_proj", hidden_states);
		ctx.Release(hidden_states);

		var keys   = BatchRelease(CacheUpdate($"{path}.k", (input_ids, 1), MarkRelease(key)));
		var values = BatchRelease(CacheUpdate($"{path}.v", (input_ids, 1), MarkRelease(value)));

		var window_size = config.attention_layers[layer_id] == "local" ? config.window_size : config.max_position_embeddings;
		var norm_factor = 1f;
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 1, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightT:true));
		hidden_states = BatchRelease(Linear($"{path}.out_proj", MarkRelease(hidden_states)));
	}
	void GPTNeoMLP(string path, ref Texture hidden_states) {
		hidden_states = BatchRelease(Linear($"{path}.c_fc", MarkRelease(hidden_states)));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.ActFn(config.hidden_act)));
		hidden_states = BatchRelease(Linear($"{path}.c_proj", MarkRelease(hidden_states)));
	}
	void GPTNeoBlock(string path, ref Texture hidden_states, Texture input_ids, int layer_id) {
		var attn_states = LayerNorm($"{path}.ln_1", hidden_states, config.layer_norm_eps);
		GPTNeoSelfAttention($"{path}.attn.attention", ref attn_states, input_ids, layer_id:layer_id);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		var mlp_states = LayerNorm($"{path}.ln_2", hidden_states, config.layer_norm_eps);
		GPTNeoMLP($"{path}.mlp", ref mlp_states);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture GPTNeoModel(string path, Texture input_ids) {
		var inputs_embeds   = Embedding($"{path}.wte", (input_ids, 0));
		var position_embeds = Embedding($"{path}.wpe", (input_ids, 1));
		var hidden_states   = BatchRelease(nn.Fusion(MarkRelease(inputs_embeds), add:MarkRelease(position_embeds)));
		for(int i=0; i<config.num_hidden_layers; i++)
			GPTNeoBlock($"{path}.h.{i}", ref hidden_states, input_ids, layer_id:i);
		hidden_states = BatchRelease(LayerNorm($"{path}.ln_f", MarkRelease(hidden_states), config.layer_norm_eps));
		return hidden_states;
	}
	(Texture, Texture) GPTNeoForCausalLM(Texture input_ids) {
		var hidden_states = GPTNeoModel("transformer", input_ids);
		var logits = Linear("lm_head", hidden_states, fallback:"transformer.wte");
		return (logits, hidden_states);
	}
}
}