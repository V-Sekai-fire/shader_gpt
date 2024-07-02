using UnityEngine;
using System.Text.RegularExpressions;

namespace ShaderGPT.Models {
[System.Serializable]
public class VitsConfig : PretrainedConfig<VitsConfig> {
	public int hidden_size;
	public int num_hidden_layers;
	public int num_attention_heads;
	public int window_size;
	public int ffn_kernel_size;
	public string hidden_act;
	public float layer_norm_eps;
	public int[] upsample_rates;
	public int[] upsample_kernel_sizes;
	public int[] resblock_kernel_sizes;
	public int[] resblock_dilation_sizes; // flattened
	public float leaky_relu_slope;
	public int prior_encoder_num_flows;
	public int prior_encoder_num_wavenet_layers;
	public int wavenet_kernel_size;
	public int wavenet_dilation_rate;
	public float noise_scale;
	public int sampling_rate;

	public static new VitsConfig FromPretrained(TextAsset configJson) {
		return JsonUtility.FromJson<VitsConfig>(Regex.Replace(Regex.Replace(Regex.Replace(
			configJson.text, @"\[\s+\[", "["), @"\]\s+\]", "]"), @"\],\s+\[", ",")); // help unity parse nested array
	}
}
public class Vits : PretrainedModel<VitsConfig> {
	public VitsEncoder text_encoder;
	public VitsFlow flow;
	public VitsDecoder decoder;
	public Vits(TensorNN nn, VitsConfig config): base(nn, config) {
		text_encoder = new VitsEncoder(nn, config){state_dict=state_dict};
		flow = new VitsFlow(nn, config){state_dict=state_dict};
		decoder = new VitsDecoder(nn, config){state_dict=state_dict};
	}
	public (Texture waveform, Texture spectrogram, Texture hidden_states) VitsModel(Texture input_ids, Texture indices,
			(Vector4,Texture)? input_padding_mask=null, (Vector4,Texture)? output_padding_mask=null) {
		var padding_mask = input_padding_mask ?? (new Vector2(0, ctx.Size0(input_ids)), default);
		var (hidden_states, stats) = text_encoder.VitsTextEncoder(input_ids, padding_mask);

		var attn_stats = BatchRelease(nn.IndexSelect(MarkRelease(stats), (indices, 0)));
		var flow_size = ctx.Size1(attn_stats)/2;
		var prior_means = ctx.Slice(attn_stats, ctx.Size0(attn_stats), flow_size);
		var prior_log_variances = ctx.Slice(attn_stats, ctx.Size0(attn_stats), flow_size, 0, flow_size);

		var prior_variances = nn.Fusion(prior_log_variances, func:TensorNN.Keyword.FUNC_EXP);
		var prior_latents = BatchRelease(nn.Fusion(prior_variances, func:TensorNN.Keyword.FUNC_NORMAL,
			mul:MarkRelease(prior_variances), scale:config.noise_scale, add:(MarkRelease(attn_stats), prior_means).Item2));
		var latents = (RenderTexture)BatchRelease(nn.Transpose(MarkRelease(prior_latents), ctx.Size1(prior_latents)*4));

		padding_mask = output_padding_mask ?? (new Vector2(0, ctx.Size0(indices)), default);
		flow.VitsResidualCouplingBlock(ref latents, padding_mask, reverse:true);
		var waveform = decoder.VitsHifiGan(latents, padding_mask);
		return (waveform, latents, hidden_states);
	}
}
}