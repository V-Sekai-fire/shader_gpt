using UnityEngine;

namespace ShaderGPT.Models {
public class VitsFlow : PretrainedModel<VitsConfig> {
	public VitsFlow(TensorNN nn, VitsConfig config): base(nn, config) {}

	void VitsWaveNet(string path, ref Texture inputs, (Vector4,Texture) padding_mask, int num_layers) {
		var hidden_size = ctx.Size0(inputs);
		var outputs = nn.Fusion(inputs, scale:0f);
		for(int i=0; i<num_layers; i++) {
			var hidden_states = Conv1d($"{path}.in_layers.{i}", inputs, config.wavenet_kernel_size,
				dilation:(int)Mathf.Pow(config.wavenet_dilation_rate, i));

			var (first_half, second_half) = Split(hidden_states, (hidden_size, hidden_size));
			var first_act  = nn.Fusion(first_half, func:TensorNN.Keyword.FUNC_TANH);
			var second_act = nn.Fusion(second_half, func:TensorNN.Keyword.FUNC_SIGMOID);
			ctx.Release(hidden_states);
			var acts = BatchRelease(nn.Fusion(MarkRelease(first_act), mul:MarkRelease(second_act), window:padding_mask));

			var res_skip_acts = BatchRelease(Conv1d($"{path}.res_skip_layers.{i}", MarkRelease(acts), 1));
			res_skip_acts = BatchRelease(nn.Fusion(MarkRelease(res_skip_acts), window:padding_mask));
			if(i < num_layers-1) {
				(first_half, second_half) = Split(res_skip_acts, (hidden_size, hidden_size));
				inputs  = BatchRelease(nn.Fusion(MarkRelease(inputs), add:first_half));
				outputs = BatchRelease(nn.Fusion(MarkRelease(outputs), add:second_half));
				ctx.Release(res_skip_acts);
			} else
				outputs = BatchRelease(nn.Fusion(MarkRelease(outputs), add:MarkRelease(res_skip_acts)));
		}
		ctx.Release(inputs);
		inputs = outputs;
	}
	void VitsResidualCouplingLayer(string path, ref RenderTexture inputs, (Vector4,Texture) padding_mask, bool reverse) {
		var half_channels = ctx.Size0(inputs)/2;
		var (first_half, second_half) = Split(inputs, (half_channels, half_channels));
		var hidden_states = Conv1d($"{path}.conv_pre", first_half, 1);
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), window:padding_mask));
		VitsWaveNet($"{path}.wavenet", ref hidden_states, padding_mask, num_layers:config.prior_encoder_num_wavenet_layers);
		var mean = BatchRelease(Conv1d($"{path}.conv_post", MarkRelease(hidden_states), 1));

		var diff = BatchRelease(nn.Fusion(MarkRelease(mean), scale:reverse?-1:+1, add:second_half, window:padding_mask));
		nn.Copy(second_half, diff);
		ctx.Release(diff);
	}
	public void VitsResidualCouplingBlock(ref RenderTexture inputs, (Vector4,Texture) padding_mask, bool reverse, string path="flow") {
		for(int i=0; i<config.prior_encoder_num_flows; i++) {
			inputs = !reverse ? inputs : (RenderTexture)BatchRelease(nn.Flip(MarkRelease(inputs)));
			VitsResidualCouplingLayer($"{path}.flows.{(reverse?config.prior_encoder_num_flows-1-i:i)}", ref inputs, padding_mask, reverse);
			inputs =  reverse ? inputs : (RenderTexture)BatchRelease(nn.Flip(MarkRelease(inputs)));
		}
	}

	(TexView,TexView) Split(Texture input, (int,int) sizes)
		=> (ctx.Slice(input, sizes.Item1, ctx.Size1(input)), ctx.Slice(input, sizes.Item2, ctx.Size1(input), sizes.Item1, 0));
}
}