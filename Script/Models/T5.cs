using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class T5Config : PretrainedConfig<T5Config> {
	public int d_model;
	public int num_layers;
	public int num_heads;
	public int relative_attention_max_distance;
	public float layer_norm_epsilon;
	public string dense_act_fn;
	public bool is_gated_act;
	public bool tie_word_embeddings;

	public int hidden_size => d_model;
	public int num_attention_heads => num_heads;
	public int num_hidden_layers => num_layers;
	public string hidden_act => dense_act_fn;
	public float rms_norm_eps => layer_norm_epsilon;
}
public class T5 : ModelForSeq2SeqLM<T5Config> {
	public T5(TensorNN nn, T5Config config): base(nn, config) {}
	public override (Texture, Texture, Texture) ForSeq2SeqLM(Texture input_ids, Texture decoder_input_ids, Texture encoder_hidden_states=null)
		=> T5ForConditionalGeneration(input_ids, decoder_input_ids, encoder_hidden_states);

	void T5Attention(string path, ref Texture hidden_states, Texture input_ids, Texture key_value_states=null,
			bool cross=false, bool causal=false) {
		var use_cache = cross ? !key_value_states : causal;
		Texture keys, values;
		if(cross && !key_value_states) { // load cross kv
			keys   = cache[$"{path}.k"];
			values = cache[$"{path}.v"];
		} else {
			var k = Linear($"{path}.k", cross ? key_value_states : hidden_states);
			var v = Linear($"{path}.v", cross ? key_value_states : hidden_states);
			if(cross && !hidden_states) { // update cross kv
				BatchRelease(CacheUpdate($"{path}.k", MarkRelease(k)));
				BatchRelease(CacheUpdate($"{path}.v", MarkRelease(v)));
				return;
			} else if(!cross && causal) { // update causal kv
				keys   = BatchRelease(CacheUpdate($"{path}.k", (input_ids, 1), MarkRelease(k)));
				values = BatchRelease(CacheUpdate($"{path}.v", (input_ids, 1), MarkRelease(v)));
			} else {
				keys   = k;
				values = v;
			}
		}
		var query  = BatchRelease(Linear($"{path}.q", MarkRelease(hidden_states)));
		var qsize0 = ctx.Size0(query);
		var ksize0 = ctx.Size0(keys);

		var attn_scores = BatchRelease(nn.Linear(MarkRelease(query), use_cache?keys:MarkRelease(keys), heads:config.num_attention_heads));
		if(!cross) {
			var window_size = config.relative_attention_max_distance;
			var position_bias = nn.Narrow(state_dict[Regex.Replace($"{path}.relative_attention_bias.weight.T", @"[.]\d+[.]", ".0.")],
				window:(new Vector4(window_size, window_size+ksize0, -1, 1), input_ids),
				groups:config.num_attention_heads, size0:qsize0, clamp:true);
			attn_scores = BatchRelease(nn.Fusion(MarkRelease(attn_scores), add:MarkRelease(position_bias)));
		}
		var attn_weights = BatchRelease(nn.Softmax(MarkRelease(attn_scores), groups:config.num_attention_heads,
			window:(causal ? new Vector4(1-ksize0, 1, 1, 1) : new Vector4(-ksize0, 0, 1, 2), input_ids)));
		hidden_states = BatchRelease(nn.Linear(MarkRelease(attn_weights), use_cache?values:MarkRelease(values), heads:config.num_attention_heads, weightT:true));
		hidden_states = BatchRelease(Linear($"{path}.o", MarkRelease(hidden_states)));
	}
	void T5DenseActDense(string path, ref Texture hidden_states) {
		if(config.is_gated_act) {
			var gate = Linear($"{path}.wi_0", hidden_states);
			var up   = Linear($"{path}.wi_1", hidden_states);
			ctx.Release(hidden_states);
			var act = BatchRelease(nn.Fusion(MarkRelease(gate), func:TensorNN.ActFn(config.hidden_act)));
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(up), mul:MarkRelease(act)));
		} else {
			hidden_states = BatchRelease(Linear($"{path}.wi", MarkRelease(hidden_states)));
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.ActFn(config.hidden_act)));
		}
		hidden_states = BatchRelease(Linear($"{path}.wo", MarkRelease(hidden_states)));
	}
	void T5Block(string path, ref Texture hidden_states, Texture input_ids, Texture encoder_hidden_states=null) {
		var is_decoder = state_dict.ContainsKey($"{path}.layer.2.layer_norm.weight");
		// when hidden_states is null, only compute cross attn kv
		if(hidden_states) {
			var attn_states = LayerNorm($"{path}.layer.0.layer_norm", hidden_states, config.rms_norm_eps, rms:true);
			T5Attention($"{path}.layer.0.SelfAttention", ref attn_states, input_ids, causal:is_decoder);
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		}
		if(is_decoder) {
			var attn_states = !hidden_states ? null : LayerNorm($"{path}.layer.1.layer_norm", hidden_states, config.rms_norm_eps, rms:true);
			T5Attention($"{path}.layer.1.EncDecAttention", ref attn_states, input_ids, encoder_hidden_states, cross:true);
			hidden_states = !hidden_states ? null : BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		}
		if(hidden_states) {
			var mlp_path = $"{path}.layer.{(is_decoder?2:1)}";
			var mlp_states = LayerNorm($"{mlp_path}.layer_norm", hidden_states, config.rms_norm_eps, rms:true);
			T5DenseActDense($"{mlp_path}.DenseReluDense", ref mlp_states);
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
		}
	}
	Texture T5Stack(string path, Texture input_ids, Texture encoder_hidden_states=null) {
		var hidden_states = !input_ids ? null : Embedding($"{path}.embed_tokens", (input_ids, 0), fallback:"shared");
		for(int i=0; i<config.num_hidden_layers; i++)
			T5Block($"{path}.block.{i}", ref hidden_states, input_ids, encoder_hidden_states);
		hidden_states = !hidden_states ? null : BatchRelease(LayerNorm($"{path}.final_layer_norm", MarkRelease(hidden_states), config.rms_norm_eps, rms:true));
		return hidden_states;
	}
	(Texture, Texture, Texture) T5ForConditionalGeneration(Texture input_ids, Texture decoder_input_ids, Texture encoder_hidden_states=null) {
		if(!encoder_hidden_states && input_ids)
			encoder_hidden_states = T5Stack("encoder", input_ids);
		var decoder_hidden_states = T5Stack("decoder", decoder_input_ids, encoder_hidden_states:encoder_hidden_states);
		var logits = !decoder_input_ids ? null : BatchRelease(Linear("lm_head", config.tie_word_embeddings ? MarkRelease(
			nn.Fusion(decoder_hidden_states, scale:1f/Mathf.Sqrt(config.hidden_size))) : decoder_hidden_states, fallback:"shared"));
		return (logits, decoder_hidden_states, encoder_hidden_states);
	}
}
}