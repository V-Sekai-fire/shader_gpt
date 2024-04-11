using UnityEngine;

namespace ShaderGPT.Models {
[System.Serializable]
public class GPT2Config {
	public int n_layer;
	public int n_head;
	public float layer_norm_epsilon;
	public string activation_function;
	public int n_positions;
}
public class GPT2 : ModelForCausalLM {
	public GPT2Config config;
	public GPT2(TensorNN nn, TextAsset configJson): base(nn, configJson) {
		config = JsonUtility.FromJson<GPT2Config>(configJson.text);
		maxLength = Mathf.Min(maxLength, config.n_positions);
	}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => GPT2LMHeadModel(input_ids);

	void GPT2Attention(ref Texture hidden_states, Texture input_ids, string path) {
		var query = nn.Linear(hidden_states, state_dict[$"{path}.c_query.weight"], state_dict[$"{path}.c_query.bias"]);
		var key   = nn.Linear(hidden_states, state_dict[$"{path}.c_key.weight"],   state_dict[$"{path}.c_key.bias"]);
		var value = nn.Linear(hidden_states, state_dict[$"{path}.c_value.weight"], state_dict[$"{path}.c_value.bias"]);
		ctx.Release(hidden_states);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", maxLength, ctx.Size1(key), dtype:ctx.DType(key));
		var values = ctx.PersistentGPUTensor($"{path}.v", maxLength, ctx.Size1(value), dtype:ctx.DType(value));
		BatchRelease(nn.IndexCopy(keys,   (input_ids, 1), MarkRelease(key)));
		BatchRelease(nn.IndexCopy(values, (input_ids, 1), MarkRelease(value)));

		var window_size = config.n_positions;
		var norm_factor = 1f / Mathf.Sqrt(ctx.Size1(query)*4 / config.n_head);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.n_head));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.n_head, window:new Vector4(1-window_size, 1, 0, 1), offset:input_ids));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.n_head, weightT:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.c_proj.weight"], state_dict[$"{path}.c_proj.bias"]));
	}
	void GPT2Block(ref Texture hidden_states, Texture input_ids, string path) {
		var attn_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.ln_1.weight"], state_dict[$"{path}.ln_1.bias"], config.layer_norm_epsilon);
		GPT2Attention(ref attn_states, input_ids, path:$"{path}.attn");
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));

		var mlp_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.ln_2.weight"], state_dict[$"{path}.ln_2.bias"], config.layer_norm_epsilon);
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), state_dict[$"{path}.mlp.c_fc.weight"], state_dict[$"{path}.mlp.c_fc.bias"]));
		mlp_states = BatchRelease(nn.Fusion(MarkRelease(mlp_states), func:TensorNN.ActFn(config.activation_function)));
		mlp_states = BatchRelease(nn.Linear(MarkRelease(mlp_states), state_dict[$"{path}.mlp.c_proj.weight"], state_dict[$"{path}.mlp.c_proj.bias"]));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture GPT2Model(Texture input_ids, string path) {
		var inputs_embeds   = nn.IndexSelect(state_dict[$"{path}.wte.weight.T"], (input_ids, 0), inputT:true);
		var position_embeds = nn.IndexSelect(state_dict[$"{path}.wpe.weight.T"], (input_ids, 1), inputT:true);
		var hidden_states   = BatchRelease(nn.Fusion(MarkRelease(inputs_embeds), add:MarkRelease(position_embeds)));
		for(int i=0; i<config.n_layer; i++)
			GPT2Block(ref hidden_states, input_ids, path:$"{path}.h.{i}");
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), state_dict[$"{path}.ln_f.weight"], state_dict[$"{path}.ln_f.bias"], config.layer_norm_epsilon));
		return hidden_states;
	}
	(Texture, Texture) GPT2LMHeadModel(Texture input_ids) {
		var hidden_states = GPT2Model(input_ids, path:"transformer");
		state_dict.TryGetValue("lm_head.weight.T", out var lm_head);
		var lm_logits = nn.Linear(hidden_states, lm_head ?? state_dict["transformer.wte.weight.T"], weightT:true);
		return (hidden_states, lm_logits);
	}
}
}