using UnityEngine;

namespace ShaderGPT.Models {
[System.Serializable]
public class GPT2Config : PretrainedConfig<GPT2Config> {
	public int n_positions;
	public int n_embd;
	public int n_layer;
	public int n_head;
	public string activation_function;
	public float layer_norm_epsilon;

	public int hidden_size => n_embd;
	public int max_position_embeddings => n_positions;
	public int num_attention_heads => n_head;
	public int num_hidden_layers => n_layer;
}
public class GPT2 : ModelForCausalLM<GPT2Config> {
	public GPT2(TensorNN nn, GPT2Config config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => GPT2LMHeadModel(input_ids);

	void GPT2Attention(string path, ref Texture hidden_states, Texture input_ids) {
		var qkv = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.c_attn.weight"], state_dict[$"{path}.c_attn.bias"]));
		var q = ctx.Slice(qkv, ctx.Size0(qkv), config.hidden_size/4);
		var k = ctx.Slice(qkv, ctx.Size0(qkv), config.hidden_size/4, 0, config.hidden_size/4);
		var v = ctx.Slice(qkv, ctx.Size0(qkv), config.hidden_size/4, 0, config.hidden_size/2);

		var keys   = ctx.PersistentGPUTensor($"{path}.k", max_length, ctx.Size1(k), dtype:ctx.DType(k));
		var values = ctx.PersistentGPUTensor($"{path}.v", max_length, ctx.Size1(v), dtype:ctx.DType(v));
		nn.IndexCopy(keys,   (input_ids, 1), k);
		nn.IndexCopy(values, (input_ids, 1), v);

		var window_size = config.max_position_embeddings;
		var norm_factor = 1f / Mathf.Sqrt(config.hidden_size / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear((MarkRelease(qkv), q).Item2, keys, heads:config.num_attention_heads)); 
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 0, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightT:true));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.c_proj.weight"], state_dict[$"{path}.c_proj.bias"]));
	}
	void GPT2MLP(string path, ref Texture hidden_states) {
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.c_fc.weight"], state_dict[$"{path}.c_fc.bias"]));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.ActFn(config.activation_function)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(hidden_states), state_dict[$"{path}.c_proj.weight"], state_dict[$"{path}.c_proj.bias"]));
	}
	void GPT2Block(string path, ref Texture hidden_states, Texture input_ids) {
		var attn_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.ln_1.weight"], state_dict[$"{path}.ln_1.bias"], config.layer_norm_epsilon);
		GPT2Attention($"{path}.attn", ref attn_states, input_ids);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		var mlp_states = nn.GroupNorm(hidden_states, state_dict[$"{path}.ln_2.weight"], state_dict[$"{path}.ln_2.bias"], config.layer_norm_epsilon);
		GPT2MLP($"{path}.mlp", ref mlp_states);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture GPT2Model(string path, Texture input_ids) {
		var inputs_embeds   = nn.IndexSelect(state_dict[$"{path}.wte.weight.T"], (input_ids, 0), inputT:true);
		var position_embeds = nn.IndexSelect(state_dict[$"{path}.wpe.weight.T"], (input_ids, 1), inputT:true);
		var hidden_states   = BatchRelease(nn.Fusion(MarkRelease(inputs_embeds), add:MarkRelease(position_embeds)));
		for(int i=0; i<config.num_hidden_layers; i++)
			GPT2Block($"{path}.h.{i}", ref hidden_states, input_ids);
		hidden_states = BatchRelease(nn.GroupNorm(MarkRelease(hidden_states), state_dict[$"{path}.ln_f.weight"], state_dict[$"{path}.ln_f.bias"], config.layer_norm_epsilon));
		return hidden_states;
	}
	(Texture, Texture) GPT2LMHeadModel(Texture input_ids) {
		var hidden_states = GPT2Model("transformer", input_ids);
		state_dict.TryGetValue("lm_head.weight.T", out var lm_head);
		var logits = nn.Linear(hidden_states, lm_head ?? state_dict["transformer.wte.weight.T"], weightT:true);
		return (hidden_states, logits);
	}
}
}