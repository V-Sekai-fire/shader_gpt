using UnityEngine;
using System.Collections.Generic;

namespace ShaderGPT {
public abstract class Module {
	protected TensorNN nn;
	protected TensorContext ctx => nn.ctx;
	public Dictionary<string, Texture> state_dict;

	public Module(TensorNN nn) {
		this.nn = nn;
		this.state_dict = new();
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
		ctx.FixSize0(state_dict[name], size0);
		if(state_dict.TryGetValue(name+".q8", out var quantizer))
			ctx.FixSize0(quantizer, (size0+3)/4);
		if(state_dict.TryGetValue(name+".q8.idx", out var permuter))
			ctx.FixSize0(permuter, 1);
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
	public PretrainedModel(TensorNN nn, T config): base(nn) {
		this.config = config;
	}
}
public interface PretrainedModel {
	public void LoadStateDict(IEnumerable<Texture> textures);
}
}