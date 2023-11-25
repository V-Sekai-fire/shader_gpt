using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public class TensorNN {
	public TensorContext ctx;
	public Dictionary<string, Shader> kernels;
	public VertexAttributeFormat dataType = VertexAttributeFormat.Float32;
	public int linearMipmap = 2;

	public Texture Embedding(Texture input, Texture weight, Texture weight2=null, bool transposeWeight=false) {
		var output = ctx.GPUTensor(ctx.Size0(input), transposeWeight ? ctx.Size0(weight)/4 : ctx.Size1(weight), dtype:dataType);
		var mat = ctx.Operator(kernels["Embedding"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(weight2)
			SetTensor(mat, "_Weight2", weight2);
		if(transposeWeight)
			EnableOption(mat, Keyword.TRANSPOSE_WEIGHT);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Linear(Texture input, Texture weight, bool transposeWeight=false, int head=1) {
		Debug.Assert(ctx.Size1(input) % head == 0 && ctx.Size1(weight) % head == 0 && ctx.Size0(weight) % 4 == 0);
		Debug.Assert(transposeWeight ? (ctx.Size0(weight)/4 == ctx.Size1(input)/head) : (ctx.Size1(weight) == ctx.Size1(input)));
		var output = ctx.GPUTensor(ctx.Size0(input), transposeWeight?ctx.Size1(weight):ctx.Size0(weight)/4*head, dtype:dataType, mipmap:linearMipmap);
		var mat = ctx.Operator(kernels["Linear"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		mat.SetInt("_Head", head);
		if(transposeWeight)
			EnableOption(mat, Keyword.TRANSPOSE_WEIGHT);
		ctx.Blit(output, mat);
		return output;
	}
	Texture _Reduce(Texture input, Keyword func, int group=1, Vector4? rangeMask=default, float seed=default, Matrix4x4? linear=default) {
		Debug.Assert(ctx.Size1(input) % group == 0);
		var output = ctx.GPUTensor(ctx.Size0(input), group, dtype:VertexAttributeFormat.Float32);
		var mat = ctx.Operator(kernels["Reduce"]);
		EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(rangeMask != default)
			mat.SetVector("_RangeMask", rangeMask.Value);
		if(seed != default)
			mat.SetFloat("_Seed", seed);
		if(linear != default) {
			mat.SetVector("_Linear0", linear.Value.GetColumn(0));
			mat.SetVector("_Linear1", linear.Value.GetColumn(1));
			mat.SetVector("_Linear2", linear.Value.GetColumn(2));
			mat.SetVector("_Linear3", linear.Value.GetColumn(3));
		}
		ctx.Blit(output, mat); 
		return output;
	}
	public Texture GroupNorm(Texture input, Texture weight, Texture bias, float eps, int group=1) {
		Debug.Assert(ctx.Size0(weight) == 1 && ctx.Size1(weight) == ctx.Size1(input));
		Debug.Assert(ctx.Size0(bias) == 1 && ctx.Size1(bias) == ctx.Size1(input));
		var reduce = _Reduce(input, Keyword.REDUCE_MOMENT, group:group);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_GROUPNORM);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Reduce", reduce);
		SetTensor(mat, "_Weight", weight);
		SetTensor(mat, "_Bias",   bias);
		mat.SetFloat("_Eps", eps);
		ctx.Blit(output, mat);
		ctx.Release(reduce);
		return output;
	}
	Texture _Normalize(Texture input, Keyword func, Keyword reduceFunc, int group=1, Vector4? rangeMask=default) {
		var reduce = _Reduce(input, reduceFunc, group:group, rangeMask:rangeMask);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Reduce", reduce);
		if(rangeMask != default)
			mat.SetVector("_RangeMask", rangeMask.Value);
		ctx.Blit(output, mat);
		ctx.Release(reduce);
		return output;
	}
	public Texture AddAct(Texture input, Texture bias=default, Keyword func=default, Vector4? weight=default) {
		Debug.Assert(!bias || (ctx.Size0(input) % ctx.Size0(bias) == 0 && ctx.Size1(input) % ctx.Size1(bias) == 0));
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		if(func != default)
			EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(bias)
			SetTensor(mat, "_Bias", bias);
		if(weight != default)
			mat.SetVector("_Weight", weight.Value);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Copy(RenderTexture output, Texture input, Vector2Int size, Vector2Int outputOffset=default, Vector2Int inputOffset=default, Vector4? weight=default, Vector4 bias=default) {
		if(object.ReferenceEquals(output, null))
			output = ctx.GPUTensor(size.x, size.y, dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output, outputOffset, size);
		SetTensor(mat, "_Input",  input, inputOffset, size);
		mat.SetVector("_Weight", weight??Vector4.one);
		mat.SetVector("_Bias",   bias);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Rotary(Texture input, Texture rotary, Vector2Int rotaryOffset=default, int group=1) {
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_ROTARY);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Rotary", rotary, rotaryOffset, ctx.Size(rotary));
		mat.SetVector("_ReduceDim", new Vector2(ctx.Size0(input), group));
		ctx.Blit(output, mat);
		return output;
	}

	public Texture ArgMax(Texture input, Vector4? rangeMask=default) {
		return _Reduce(input, Keyword.REDUCE_MINMAX, rangeMask:rangeMask,
			linear:new Matrix4x4(default, default, default, new Vector4(1,0,0,0)));
	}
	public Texture Softmax(Texture input, int group=1, Vector4? rangeMask=default) {
		var temp = _Normalize(input, Keyword.FUNC_SOFTMAX_LINF, Keyword.REDUCE_MINMAX, group:group, rangeMask:rangeMask);
		var output = _Normalize(temp, Keyword.FUNC_NORMALIZE_L1, Keyword.REDUCE_MOMENT, group:group);
		ctx.Release(temp);
		return output;
	}
	public Texture Gumbel(Texture input, float temperature) {
		return AddAct(input, func:Keyword.FUNC_GUMBEL, weight:Vector4.one*temperature, bias:input);
	}

	void SetTensor(Material mat, string name, Texture tex) {
		mat.SetTexture($"{name}Tex", tex);
		mat.SetVector($"{name}Dim",  new Vector4(ctx.Size0(tex), ctx.Size1(tex), 1, ctx.Mipmap(tex)));
	}
	void SetTensor(Material mat, string name, Texture tex, Vector2Int offset, Vector2Int size) {
		SetTensor(mat, name, tex);
		mat.SetVector($"{name}Dim",  new Vector4(size.x, size.y, 1, ctx.Mipmap(tex)));
		mat.SetVector($"{name}Off",  new Vector4(offset.x, offset.y, 0, 0));
	}
	void EnableOption(Material mat, Keyword keyword) {
		mat.EnableKeyword(keyword.ToString());
	}
	static public Keyword ActFn(string name) {
		return (Keyword)System.Enum.Parse(typeof(Keyword), $"FUNC_{name.ToUpperInvariant()}");
	}

	public enum Keyword {
		None = 0,
		TRANSPOSE_WEIGHT,
		REDUCE_MOMENT,
		REDUCE_MINMAX,
		FUNC_GROUPNORM,
		FUNC_SOFTMAX_LINF,
		FUNC_NORMALIZE_L1,
		FUNC_GELU,
		FUNC_GELU_NEW,
		FUNC_GUMBEL,
		FUNC_ROTARY,
	}
}
}