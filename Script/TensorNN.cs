using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public class TensorNN {
	public TensorContext ctx;
	public Dictionary<string, Shader> kernels;
	public Dictionary<Texture, Texture> quants = new Dictionary<Texture, Texture>();
	public VertexAttributeFormat dataType = VertexAttributeFormat.Float32;
	public int linearMipmap = 2; // 3 has higher occupancy but similar gpu time
	public int reduceSplit = 64;

	public Texture Embedding(Texture input, Texture weight0, Texture weight1=null, bool transposeWeight=false) {
		var output = ctx.GPUTensor(ctx.Size0(input), transposeWeight ? ctx.Size0(weight0??weight1)/4 : ctx.Size1(weight0??weight1), dtype:dataType);
		var mat = ctx.Operator(kernels["Embedding"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(weight0) {
			SetTensor(mat, "_Weight0", weight0);
			if(quants.TryGetValue(weight0, out var quant0)) {
				SetTensor(mat, "_Scale0", quant0);
				EnableOption(mat, Keyword.WEIGHT_QUANTIZED);
			}
		}
		if(weight1) {
			SetTensor(mat, "_Weight1", weight1);
			if(quants.TryGetValue(weight1, out var quant1)) {
				SetTensor(mat, "_Scale1", quant1);
				EnableOption(mat, Keyword.WEIGHT_QUANTIZED);
			}
		}
		if(transposeWeight)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Linear(Texture input, Texture weight, bool transposeWeight=false, int heads=1, int weightHeads=0, float scale=1f) {
		if(weightHeads == 0)
			weightHeads = heads;
		Debug.Assert(ctx.Size1(input)%heads == 0 && ctx.Size1(weight)%weightHeads == 0 && ctx.Size0(weight)%4 == 0 && heads%weightHeads == 0);
		Debug.Assert(ctx.Size1(input)/heads == (transposeWeight ? ctx.Size0(weight)/4 : ctx.Size1(weight)/weightHeads));
		var size1 = (transposeWeight ? ctx.Size1(weight)/weightHeads : ctx.Size0(weight)/4) * heads;
		var output = ctx.GPUTensor(ctx.Size0(input), size1, dtype:dataType, mipmap:linearMipmap,
			autoMips:weight is RenderTexture); // NOTE: not fully correct but works in common cases
		var mat = ctx.Operator(kernels["Linear"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(quants.TryGetValue(weight, out var quant)) {
			EnableOption(mat, Keyword.WEIGHT_QUANTIZED);
			SetTensor(mat, "_Scale", quant);
		}
		if(transposeWeight)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		mat.SetFloat("_Scale", scale);
		ctx.Blit(output, mat);
		return output;
	}
	Texture _Reduce(Texture input, Keyword func, int groups=1, Vector4? indexRange=default, Texture rangeOffset=default, Matrix4x4? linear=default, bool inputReduced=false) {
		Debug.Assert(ctx.Size1(input) % groups == 0);
		var size1 = groups;
		if(!inputReduced && ctx.Size1(input) / groups >= reduceSplit) {
			var n = ctx.Size1(input) / groups;
			if(groups == 1)
				size1 = Mathf.CeilToInt(Mathf.Sqrt(n));
			else { // ensure ctx.Size1(input) % size1 == 0 && size1 % groups == 0
				var m = 1 << Mathf.CeilToInt(Mathf.Log(n, 2)/2);
				if(n % m == 0)
					size1 = groups * m;
			}
		}
		var output = ctx.GPUTensor(ctx.Size0(input), size1, dtype:VertexAttributeFormat.Float32);
		var mat = ctx.Operator(kernels["Reduce"]);
		EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(inputReduced)
			EnableOption(mat, Keyword.INPUT_REDUCED);
		if(indexRange != default)
			mat.SetVector("_IndexRange", indexRange.Value);
		if(rangeOffset)
			SetTensor(mat, "_Offset", rangeOffset);
		if(size1 > groups) {
			mat.SetInt("_IndexMod", ctx.Size1(input) / groups);
			ctx.Blit(output, mat);
			var output2 = _Reduce(output, func, groups:groups, linear:linear, inputReduced:true);
			ctx.Release(output);
			return output2;
		}
		if(linear != default) {
			mat.SetVector("_Linear0", linear.Value.GetColumn(0));
			mat.SetVector("_Linear1", linear.Value.GetColumn(1));
			mat.SetVector("_Linear2", linear.Value.GetColumn(2));
			mat.SetVector("_Linear3", linear.Value.GetColumn(3));
		}
		ctx.Blit(output, mat);
		return output;
	}
	public Texture GroupNorm(Texture input, Texture weight, Texture bias, float eps, int groups=1, bool rmsNorm=false) {
		Debug.Assert(ctx.Size0(weight) == 1 && ctx.Size1(weight) == ctx.Size1(input));
		Debug.Assert(rmsNorm ? !bias : (ctx.Size0(bias) == 1 && ctx.Size1(bias) == ctx.Size1(input)));
		var reduce = _Reduce(input, Keyword.REDUCE_SUMPOW, groups:groups,
			linear: rmsNorm ? (Matrix4x4?)new Matrix4x4(new Vector4(1,0,0,0), default, new Vector4(0,0,1,0), default) : null);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_GROUPNORM);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Reduce", reduce);
		if(weight)
			SetTensor(mat, "_Weight", weight);
		if(bias)
			SetTensor(mat, "_Bias",   bias);
		mat.SetFloat("_Eps", eps);
		ctx.Blit(output, mat);
		ctx.Release(reduce);
		return output;
	}
	Texture _Normalize(Texture input, Keyword func, Keyword reduceFunc, int groups=1, Vector4? indexRange=default, Texture rangeOffset=default) {
		var reduce = _Reduce(input, reduceFunc, groups:groups, indexRange:indexRange, rangeOffset:rangeOffset);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Reduce", reduce);
		if(indexRange != default)
			mat.SetVector("_IndexRange", indexRange.Value);
		if(rangeOffset)
			SetTensor(mat, "_Offset", rangeOffset);
		ctx.Blit(output, mat);
		ctx.Release(reduce);
		return output;
	}
	public Texture Fusion(Texture input, Vector4? scale=default, Texture mul=default, Texture add=default, Keyword func=default) {
		Debug.Assert(!mul || (ctx.Size0(input) % ctx.Size0(mul) == 0 && ctx.Size1(input) % ctx.Size1(mul) == 0));
		Debug.Assert(!add || (ctx.Size0(input) % ctx.Size0(add) == 0 && ctx.Size1(input) % ctx.Size1(add) == 0));
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(scale != default)
			mat.SetVector("_Weight", scale.Value);
		if(mul)
			SetTensor(mat, "_Weight", mul);
		if(add)
			SetTensor(mat, "_Bias", add);
		if(func != default)
			EnableOption(mat, func);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Copy(RenderTexture output, Texture input, Vector2Int size, Vector2Int outputOffset=default, Vector2Int inputOffset=default) {
		if(object.ReferenceEquals(output, null))
			output = ctx.GPUTensor(size.x, size.y, dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output, outputOffset, size);
		SetTensor(mat, "_Input",  input, inputOffset, size);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Scatter(RenderTexture output, Texture index, Texture src, Vector2 indexMask) {
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  src);
		SetTensor(mat, "_Offset", index);
		mat.SetVector("_OutputOff",  new Vector4(0, 0, indexMask.x, indexMask.y));
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Rotary(Texture input, Texture rotary, int groups=1) {
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_ROTARY);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Rotary", rotary);
		mat.SetVector("_ReduceDim", new Vector2(ctx.Size0(input), groups));
		ctx.Blit(output, mat);
		return output;
	}

	public Texture ArgMax(Texture input, Vector4? indexRange=default) {
		return _Reduce(input, Keyword.REDUCE_MINMAX, indexRange:indexRange,
			linear:new Matrix4x4(default, default, default, new Vector4(1,0,0,0)));
	}
	public Texture Softmax(Texture input, int groups=1, Vector4? indexRange=default, Texture rangeOffset=default) {
		var temp = _Normalize(input, Keyword.FUNC_SOFTMAX_LINF, Keyword.REDUCE_MINMAX, groups:groups, indexRange:indexRange, rangeOffset:rangeOffset);
		var output = _Normalize(temp, Keyword.FUNC_NORMALIZE_L1, Keyword.REDUCE_SUMPOW, groups:groups);
		ctx.Release(temp);
		return output;
	}
	public Texture Gumbel(Texture input, float temperature) {
		return Fusion(input, func:Keyword.FUNC_GUMBEL, scale:Vector4.one*temperature, add:input);
	}

	void SetTensor(Material mat, string name, Texture tex) {
		mat.SetTexture($"{name}Tex", tex);
		mat.SetVector($"{name}Dim",  new Vector4(ctx.Size0(tex), ctx.Size1(tex), ctx.Wrap1(tex), ctx.Mipmap(tex)));
	}
	void SetTensor(Material mat, string name, Texture tex, Vector2Int offset, Vector2Int size) {
		SetTensor(mat, name, tex);
		mat.SetVector($"{name}Dim",  new Vector4(size.x, size.y, ctx.Wrap1(tex), ctx.Mipmap(tex)));
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
		WEIGHT_TRANSPOSED,
		WEIGHT_QUANTIZED,
		INPUT_REDUCED,
		REDUCE_SUMPOW,
		REDUCE_MINMAX,
		FUNC_GROUPNORM,
		FUNC_SOFTMAX_LINF,
		FUNC_NORMALIZE_L1,
		FUNC_GUMBEL,
		FUNC_ROTARY,
		FUNC_GELU,
		FUNC_GELU_NEW,
		FUNC_SILU,
	}
}
}