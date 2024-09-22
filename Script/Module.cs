using UnityEngine;
using System.Collections.Generic;

namespace ShaderGPT {
public abstract class Module {
	protected TensorNN nn;
	protected TensorContext ctx => nn.ctx;
	public Dictionary<string, Texture> state_dict;

	public Module(TensorNN nn) {
		this.nn = nn;
		this.state_dict = new Dictionary<string, Texture>();
	}
	public void LoadStateDict(IEnumerable<Texture> textures) {
		foreach(var tex in textures)
			state_dict[tex.name] = tex;
		foreach(var tex in textures) {
			if(state_dict.TryGetValue(tex.name+".q8", out var quantizer))
				nn.quantizers[tex] = quantizer;
			if(state_dict.TryGetValue(tex.name+".q8.idx", out var permuter))
				nn.permuters[tex] = permuter;
		}
	}

	// utilities
	List<Texture> releaseList = new List<Texture>();
	protected T MarkRelease<T>(T tex) where T: Texture {
		releaseList.Add(tex);
		return tex;
	}
	protected T BatchRelease<T>(T x) {
		foreach(var tex in releaseList)
			ctx.Release(tex);
		releaseList.Clear();
		return x;
	}
	protected void FixSize0(string name, int size0) {
		if(state_dict.TryGetValue(name, out var tensor)) // in case a fallback is used
			ctx.FixSize0(tensor, size0);
		if(state_dict.TryGetValue(name+".q8", out var quantizer))
			ctx.FixSize0(quantizer, (size0+3)/4);
		if(state_dict.TryGetValue(name+".q8.idx", out var permuter))
			ctx.FixSize0(permuter, 2);
	}

	// common layers
	protected Texture Embedding(string path, (TexView, int) input, string fallback=null) {
		if(fallback != null && !state_dict.ContainsKey($"{path}.weight.T") && !state_dict.ContainsKey($"{path}.weight"))
			path = fallback;
		state_dict.TryGetValue($"{path}.weight.T", out var weightT);
		return nn.IndexSelect(weightT ?? state_dict[$"{path}.weight"], input, inputT:weightT);
	}
	protected Texture Linear(string path, TexView input, string fallback=null) {
		if(fallback != null && !state_dict.ContainsKey($"{path}.weight.T") && !state_dict.ContainsKey($"{path}.weight"))
			path = fallback;
		state_dict.TryGetValue($"{path}.weight.T", out var weightT);
		state_dict.TryGetValue($"{path}.bias", out var bias);
		return nn.Linear(input, weightT ?? state_dict[$"{path}.weight"], bias, weightT:weightT);
	}
	protected Texture Conv1d(string path, TexView input, int kernel_size, int stride=1, int dilation=1) {
		state_dict.TryGetValue($"{path}.bias", out var bias);
		return nn.Conv1d(input, state_dict[$"{path}.weight"], bias, kernel_size, stride:stride, dilation:dilation);
	}
	protected Texture ConvTranspose1d(string path, TexView input, int kernel_size, int stride=1) {
		state_dict.TryGetValue($"{path}.bias", out var bias);
		return nn.ConvTranspose1d(input, state_dict[$"{path}.weight"], bias, kernel_size, stride:stride);
	}
	protected Texture LayerNorm(string path, TexView input, float eps, int groups=1, bool rms=false) {
		state_dict.TryGetValue($"{path}.bias", out var bias);
		return nn.GroupNorm(input, state_dict[$"{path}.weight"], bias, eps, groups:groups, rms:rms);
	}
}
[System.Serializable]
public class PretrainedConfig {
	public string model_type;
	public int vocab_size;
}
public abstract class PretrainedConfig<S> : PretrainedConfig where S : PretrainedConfig<S> {
	public static S FromPretrained(TextAsset configJson) {
		return JsonUtility.FromJson<S>(configJson.text);
	}
}
public abstract class PretrainedModel<T> : Module, PretrainedModel where T : PretrainedConfig<T> {
	public T config;
	public string model_type => config.model_type;
	public PretrainedModel(TensorNN nn, T config): base(nn) {
		this.config = config;
	}
}
public interface PretrainedModel {
	/*public*/ string model_type {get;}
	/*public*/ void LoadStateDict(IEnumerable<Texture> textures);
}
}