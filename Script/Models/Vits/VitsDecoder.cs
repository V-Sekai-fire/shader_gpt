using UnityEngine;

namespace ShaderGPT.Models {
public class VitsDecoder : PretrainedModel<VitsConfig> {
	public VitsDecoder(TensorNN nn, VitsConfig config): base(nn, config) {}

	int num_kernels => config.resblock_kernel_sizes.Length;
	int num_dilations => config.resblock_dilation_sizes.Length / config.resblock_kernel_sizes.Length;
	int num_upsamples => config.upsample_rates.Length;

	Texture HifiGanResidualBlock(string path, Texture hidden_states, (Vector4,Texture) padding_mask, int block_id) {
		var kernel_size = config.resblock_kernel_sizes[block_id];
		for(int i=0; i<num_dilations; i++) {
			var dilation = config.resblock_dilation_sizes[block_id*num_dilations+i];
			var residual = hidden_states;
			hidden_states = LeakyRelu(hidden_states, config.leaky_relu_slope);
			hidden_states = BatchRelease(Conv1d($"{path}.convs1.{i}", MarkRelease(hidden_states), kernel_size, dilation:dilation));
			hidden_states = BatchRelease(LeakyRelu(MarkRelease(hidden_states), config.leaky_relu_slope, window:padding_mask));
			hidden_states = BatchRelease(Conv1d($"{path}.convs2.{i}", MarkRelease(hidden_states), kernel_size));
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), add:residual, window:padding_mask));
			if(i>0)
				ctx.Release(residual);
		}
		return hidden_states;
	}
	public Texture VitsHifiGan(Texture spectrogram, (Vector4,Texture) padding_mask, string path="decoder") {
		var hidden_states = nn.Fusion(spectrogram, window:padding_mask); // truncate input
		hidden_states = BatchRelease(Conv1d($"{path}.conv_pre", MarkRelease(hidden_states), 7));
		hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), window:padding_mask));
		for(int i=0; i<num_upsamples; i++) {
			hidden_states = BatchRelease(LeakyRelu(MarkRelease(hidden_states), config.leaky_relu_slope));
			hidden_states = BatchRelease(ConvTranspose1d($"{path}.upsampler.{i}", MarkRelease(hidden_states),
				config.upsample_kernel_sizes[i], stride:config.upsample_rates[i]));
			padding_mask.Item1.x *= config.upsample_rates[i];
			padding_mask.Item1.y *= config.upsample_rates[i];
			padding_mask.Item1.z *= config.upsample_rates[i];
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(hidden_states), window:padding_mask));

			var res_state = default(Texture);
			for(int j=0; j<num_kernels; j++) {
				var x = HifiGanResidualBlock($"{path}.resblocks.{i*num_kernels+j}", hidden_states, padding_mask, block_id:j);
				res_state = j == 0 ? x : BatchRelease(nn.Fusion(MarkRelease(res_state), add:MarkRelease(x)));
			}
			ctx.Release(hidden_states);
			hidden_states = BatchRelease(nn.Fusion(MarkRelease(res_state), scale:1f/num_kernels));
		}
		hidden_states = BatchRelease(LeakyRelu(MarkRelease(hidden_states)));
		hidden_states = BatchRelease(Conv1d($"{path}.conv_post", MarkRelease(hidden_states), 7));
		var waveform = BatchRelease(nn.Fusion(MarkRelease(hidden_states), func:TensorNN.Keyword.FUNC_TANH, window:padding_mask));
		return waveform;
	}

	Texture LeakyRelu(TexView input, float eps=0.01f, (Vector4,Texture)? window=null)
		=> nn.Fusion(input, func:TensorNN.Keyword.FUNC_RELU, eps:eps, window:window);
}
}