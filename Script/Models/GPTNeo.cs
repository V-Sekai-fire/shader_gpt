using UnityEngine;

namespace ShaderGPT.Models {
[System.Serializable]
public class GPTNeoConfig : PretrainedConfig {
	public int max_position_embeddings;
	public int hidden_size;
	public int num_layers;
	public int num_heads;
	public int window_size;
	public string activation_function;
	public float layer_norm_epsilon;

	public int num_attention_heads => num_heads;
	public int num_hidden_layers => num_layers;
}
public class GPTNeo : ModelForCausalLM<GPTNeoConfig> {
	public GPTNeo(TensorNN nn, TextAsset configJson): base(nn, configJson) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => GPTNeoForCausalLM(input_ids);

	void GPTNeoSelfAttention(ref Texture hidden_states, Texture input_ids, string path, int layer_id) {
		var query = nn.Linear(hidden_states, state_dict[$"{path}.q_proj.weight"]);
		var key   = nn.Linear(hidden_states, state_dict[$"{path}.k_proj.weight"]);
		var value = nn.Linear(hidden_states, state_dict[$"{path}.v_proj.weight"]);
		ctx.Release(hidden_states);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", max_length, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", max_length, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), MarkRelease(value)));

		var window_size = layer_id%2 == 1 ? config.window_size : config.max_position_embeddings;
		var norm_factor = 1f;
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 0, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightT:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.out_proj.weight"], state_dict[$"{path}.out_proj.bias"]));
	}
	void GPTNeoMLP(ref Texture hidden_states, string path) {
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.c_fc.weight"], state_dict[$"{path}.c_fc.bias"]));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.ActFn(config.activation_function)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.c_proj.weight"], state_dict[$"{path}.c_proj.bias"]));
	}
	void GPTNeoBlock(ref Texture hidden_states, Texture input_ids, string path, int layer_id) {
		var attn_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.ln_1.weight"], state_dict[$"{path}.ln_1.bias"], config.layer_norm_epsilon);
		GPTNeoSelfAttention(ref attn_states, input_ids, path:$"{path}.attn.attention", layer_id:layer_id);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		var mlp_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.ln_2.weight"], state_dict[$"{path}.ln_2.bias"], config.layer_norm_epsilon);
		GPTNeoMLP(ref mlp_states, path:$"{path}.mlp");
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture GPTNeoModel(Texture input_ids, string path) {
		var inputs_embeds   = nn.IndexSelect(state_dict[$"{path}.wte.weight.T"], (input_ids, 0), inputT:true);
		var position_embeds = nn.IndexSelect(state_dict[$"{path}.wpe.weight.T"], (input_ids, 1), inputT:true);
		var hidden_states   = BatchRelease(nn.Fusion(MarkRelease(inputs_embeds), add:MarkRelease(position_embeds)));
		for(int i=0; i<config.num_hidden_layers; i++)
			GPTNeoBlock(ref hidden_states, input_ids, path:$"{path}.h.{i}", layer_id:i);
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), state_dict[$"{path}.ln_f.weight"], state_dict[$"{path}.ln_f.bias"], config.layer_norm_epsilon));
		return hidden_states;
	}
	(Texture, Texture) GPTNeoForCausalLM(Texture input_ids) {
		var hidden_states = GPTNeoModel(input_ids, path:"transformer");
		state_dict.TryGetValue("lm_head.weight.T", out var lm_head);
		var logits = nn.Linear(hidden_states, lm_head ?? state_dict["transformer.wte.weight.T"], weightT:true);
		return (hidden_states, logits);
	}
}
}