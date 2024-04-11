using UnityEngine;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public abstract class Module {
	protected Dictionary<string, Texture> state_dict;
	protected TensorNN nn;
	protected TensorContext ctx {
		get => nn.ctx;
		set { nn.ctx = value; }
	}

	public Module(TensorNN nn) {
		this.nn = nn;
	}
	public void LoadStateDict(Texture[] textures) {
		state_dict = textures.ToDictionary(x => x.name, x => x);
		foreach(var pair in state_dict) {
			if(state_dict.TryGetValue(pair.Key+".q8", out var quantizer))
				nn.quantizers[pair.Value] = quantizer;
			if(state_dict.TryGetValue(pair.Key+".q8.idx", out var permuter))
				nn.permuters[pair.Value] = permuter;
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
	}
}
}