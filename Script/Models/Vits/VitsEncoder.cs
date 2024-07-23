using UnityEngine;

namespace ShaderGPT.Models {
public class VitsEncoder : PretrainedModel<VitsConfig> {
	public VitsEncoder(TensorNN nn, VitsConfig config): base(nn, config) {}

	Texture VitsAttention(string path, Texture hidden_states, (Vector4,Texture) padding_mask, Texture input_ids) {
		var window_size = config.window_size;
		var size0 = ctx.Size0(hidden_states);
		var query  = Linear($"{path}.q_proj", hidden_states);
		var keys   = Linear($"{path}.k_proj", hidden_states);
		var values = Linear($"{path}.v_proj", hidden_states);
		
		var attn_weights = BatchRelease(nn.Linear(query, MarkRelease(keys), heads:config.num_attention_heads));
		if(window_size > 0) {
			var relative_logits = BatchRelease(nn.Linear(MarkRelease(query), state_dict[$"{path}.emb_rel_k.weight"], heads:config.num_attention_heads, weightHeads:1));
			var rel_pos_bias = BatchRelease(nn.Narrow(MarkRelease(relative_logits),
				window:(new Vector4(window_size, window_size+size0, -1, 1), input_ids), groups:config.num_attention_heads));
			Debug.Assert(ctx.Size1(rel_pos_bias) == ctx.Size1(attn_weights));
			// NOTE: rel_pos_bias will be truncated at padding_mask later in attn_probs
			attn_weights = BatchRelease(nn.Fusion(MarkRelease(attn_weights), add:MarkRelease(rel_pos_bias)));
		} else
			ctx.Release(query);

		var norm_factor = 1f / Mathf.Sqrt(config.hidden_size / config.num_attention_heads);
		var attn_probs = BatchRelease(nn.Softmax(MarkRelease(attn_weights), scale:norm_factor,
			window:padding_mask, groups:config.num_attention_heads));

		hidden_states = BatchRelease(nn.Linear(attn_probs, MarkRelease(values), weightT:true, heads:config.num_attention_heads));
		if(window_size > 0) {
			var relative_weights = BatchRelease(nn.Narrow(MarkRelease(attn_probs),
				window:(new Vector4(-window_size, window_size+1, 1, 1), input_ids), groups:config.num_attention_heads));
			Debug.Assert(ctx.Size1(relative_weights) == (window_size/2+1)*config.num_attention_heads);
			var rel_pos_bias = BatchRelease(nn.Linear(MarkRelease(relative_weights), state_dict[$"{path}.emb_rel_v.weight"], weightT:true, heads:config.num_attention_heads, weightHeads:1));
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(rel_pos_bias)));
		} else
			ctx.Release(attn_probs);

		hidden_states = BatchRelease(Linear($"{path}.out_proj", MarkRelease(hidden_states)));
		return hidden_states;
	}
	Texture VitsFeedForward(string path, Texture hidden_states, (Vector4,Texture) padding_mask) {
		var size0 = ctx.Size0(hidden_states);
		hidden_states = nn.Transpose(hidden_states, size0:ctx.Size1(hidden_states)*4);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), window:padding_mask)); // truncate input
		hidden_states = BatchRelease(Conv1d($"{path}.conv_1", MarkRelease(hidden_states), config.ffn_kernel_size));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), window:padding_mask, func:TensorNN.ActFn(config.hidden_act)));
		hidden_states = BatchRelease(Conv1d($"{path}.conv_2", MarkRelease(hidden_states), config.ffn_kernel_size));
		hidden_states = BatchRelease(nn.Transpose(MarkRelease(hidden_states), size0:size0));
		return hidden_states;
	}
	void VitsEncoderLayer(string path, ref Texture hidden_states, (Vector4,Texture) padding_mask, Texture input_ids) {
		var attn_states = VitsAttention($"{path}.attention", hidden_states, padding_mask, input_ids);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(attn_states)));
		hidden_states = BatchRelease(LayerNorm($"{path}.layer_norm", MarkRelease(hidden_states), config.layer_norm_eps));
		var mlp_states = VitsFeedForward($"{path}.feed_forward", hidden_states, padding_mask);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:MarkRelease(mlp_states)));
		hidden_states = BatchRelease(LayerNorm($"{path}.final_layer_norm", MarkRelease(hidden_states), config.layer_norm_eps));
	}
	public (Texture, Texture) VitsTextEncoder(Texture input_ids, (Vector4,Texture) padding_mask, string path="text_encoder") {
		var inputs_embeds = Embedding($"{path}.embed_tokens", (input_ids, 0));
		var hidden_states = BatchRelease(nn.Fusion(MarkRelease(inputs_embeds), scale:Mathf.Sqrt(config.hidden_size)));
		for(int i=0; i<config.num_hidden_layers; i++)
			VitsEncoderLayer($"{path}.encoder.layers.{i}", ref hidden_states, padding_mask, input_ids);
		var stats = Linear($"{path}.project", hidden_states);
		return (hidden_states, stats);
	}
}
}