using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class LlamaConfig : PretrainedConfig<LlamaConfig> {
	public int max_position_embeddings;
	public int hidden_size;
	public int num_hidden_layers;
	public int num_attention_heads;
	public int num_key_value_heads;
	public string hidden_act;
	public float rms_norm_eps;

	public string hidden_activation; // for gemma
	public float layer_norm_eps; // for stablelm
	public int sliding_window; // for mistral & qwen2
}
public class Llama : ModelForCausalLM<LlamaConfig> {
	public Llama(TensorNN nn, LlamaConfig config): base(nn, config) {}
	public override (Texture, Texture) ForCausalLM(Texture input_ids) => LlamaForCausalLM(input_ids);

	string hidden_act => config.model_type != "gemma" ? config.hidden_act
		: string.IsNullOrEmpty(config.hidden_activation) ? "gelu_pytorch_tanh" : config.hidden_activation;
	float norm_eps => config.layer_norm_eps == 0f ? config.rms_norm_eps : config.layer_norm_eps;
	bool rms => config.layer_norm_eps == 0f;

	void LlamaAttention(string path, ref Texture hidden_states, Texture input_ids) {
		TexView q, k, v;
		var merged = state_dict.ContainsKey($"{path}.qkv_proj.weight");
		if(merged) {
			var kv_hidden_size = config.hidden_size / config.num_attention_heads * config.num_key_value_heads;
			var qkv = Linear($"{path}.qkv_proj", hidden_states);
			q = ctx.Slice(qkv, ctx.Size0(qkv), config.hidden_size/4);
			k = ctx.Slice(qkv, ctx.Size0(qkv), kv_hidden_size/4, 0, config.hidden_size/4);
			v = ctx.Slice(qkv, ctx.Size0(qkv), kv_hidden_size/4, 0, config.hidden_size/4+kv_hidden_size/4);
		} else {
			q = Linear($"{path}.q_proj", hidden_states);
			k = Linear($"{path}.k_proj", hidden_states);
			v = Linear($"{path}.v_proj", hidden_states);
		}
		ctx.Release(hidden_states);

		var rotary = Embedding($"{path}.rotary_emb", (input_ids, 1), fallback:Regex.Replace($"{path}.rotary_emb", @"[.]\d+[.]", ".0."));
		var query  = BatchRelease(nn.Rotary(merged ? q : MarkRelease((Texture)q), rotary, groups:config.num_attention_heads));
		var key    = BatchRelease(nn.Rotary(merged ? k : MarkRelease((Texture)k), rotary, groups:config.num_key_value_heads));
		ctx.Release(rotary);
		var keys   = BatchRelease(CacheUpdate($"{path}.k", (input_ids, 1), MarkRelease(key)));
		var values = BatchRelease(CacheUpdate($"{path}.v", (input_ids, 1), (MarkRelease((Texture)v), v).Item2));

		var window_size = config.sliding_window == 0 ? config.max_position_embeddings : config.sliding_window;
		var norm_factor = 1f / Mathf.Sqrt(config.hidden_size / config.num_attention_heads);
		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), keys, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads));
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), scale:norm_factor,
			groups:config.num_attention_heads, window:(new Vector4(1-window_size, 1, 1, 1), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), values, heads:config.num_attention_heads, weightHeads:config.num_key_value_heads, weightT:true));
		hidden_states = BatchRelease(Linear($"{path}.o_proj", MarkRelease(hidden_states)));
	}
	void LlamaMLP(string path, ref Texture hidden_states) {
		TexView gate, up;
		var merged = state_dict.ContainsKey($"{path}.gate_up_proj.weight");
		if(merged) {
			var gate_up = Linear($"{path}.gate_up_proj", hidden_states);
			gate = ctx.Slice(gate_up, ctx.Size0(gate_up), ctx.Size1(gate_up)/2);
			up   = ctx.Slice(gate_up, ctx.Size0(gate_up), ctx.Size1(gate_up)/2, 0, ctx.Size1(gate_up)/2);
		} else {
			gate = Linear($"{path}.gate_proj", hidden_states);
			up   = Linear($"{path}.up_proj",   hidden_states);
		}
		ctx.Release(hidden_states);

		var act = BatchRelease(nn.Fusion(merged ? gate : MarkRelease((Texture)gate), func:TensorNN.ActFn(hidden_act)));
		hidden_states = BatchRelease(nn.Fusion((MarkRelease((Texture)up), up).Item2, mul:MarkRelease(act)));
		hidden_states = BatchRelease(Linear($"{path}.down_proj", MarkRelease(hidden_states)));
	}
	void LlamaDecoderLayer(string path, ref Texture hidden_states, Texture input_ids, float scale=1f) {
		var attn_states = LayerNorm($"{path}.input_layernorm", hidden_states, norm_eps, rms:rms);
		LlamaAttention($"{path}.self_attn", ref attn_states, input_ids);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), scale:scale, add:MarkRelease(attn_states)));
		var mlp_states = LayerNorm($"{path}.post_attention_layernorm", hidden_states, norm_eps, rms:rms);
		LlamaMLP($"{path}.mlp", ref mlp_states);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
	}
	Texture LlamaModel(string path, Texture input_ids) {
		FixSize0($"{path}.embed_tokens.weight.T", config.hidden_size);
		var hidden_states = Embedding($"{path}.embed_tokens", (input_ids, 0));
		for(int i=0; i<config.num_hidden_layers; i++)
			LlamaDecoderLayer($"{path}.layers.{i}", ref hidden_states, input_ids,
				scale:(i == 0 && config.model_type == "gemma" ? Mathf.Sqrt(config.hidden_size) : 1f));
		hidden_states = BatchRelease(LayerNorm($"{path}.norm", MarkRelease(hidden_states), norm_eps, rms:rms));
		return hidden_states;
	}
	(Texture, Texture) LlamaForCausalLM(Texture input_ids) {
		FixSize0("lm_head.weight.T", config.hidden_size);
		var hidden_states = LlamaModel("model", input_ids);
		var logits = Linear("lm_head", hidden_states, fallback:"model.embed_tokens");
		return (logits, hidden_states);
	}
}
}