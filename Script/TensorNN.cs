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
	public int linearLod = 2; // 3 has higher occupancy but similar gpu time
	public int reduceSplit = 64;

	public Texture Embedding(Texture input, Texture weight, bool transposeWeight=false, int chan=0) {
		var output = ctx.GPUTensor(ctx.Size0(input), transposeWeight ? ctx.Size0(weight)/4 : ctx.Size1(weight), dtype:dataType);
		var mat = ctx.Operator(kernels["Embedding"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(quants.TryGetValue(weight, out var quant))
			SetTensorQuant(mat, "_Scale", quant);
		if(transposeWeight)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		mat.SetVector("_InputChan", new Vector4(chan==0?1:0, chan==1?1:0, chan==2?1:0, chan==3?1:0));
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Linear(Texture input, Texture weight, bool transposeWeight=false, int heads=1, int weightHeads=0) {
		if(weightHeads == 0)
			weightHeads = heads;
		Debug.Assert(ctx.Size1(input)%heads == 0 && ctx.Size1(weight)%weightHeads == 0 && ctx.Size0(weight)%4 == 0 && heads%weightHeads == 0);
		Debug.Assert(ctx.Size1(input)/heads == (transposeWeight ? ctx.Size0(weight)/4 : ctx.Size1(weight)/weightHeads));
		var size1 = (transposeWeight ? ctx.Size1(weight)/weightHeads : ctx.Size0(weight)/4) * heads;
		var output = ctx.GPUTensor(ctx.Size0(input), size1, dtype:dataType, lod:linearLod,
			genMips:weight is RenderTexture); // NOTE: not fully correct but works in common cases
		var mat = ctx.Operator(kernels["Linear"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(quants.TryGetValue(weight, out var quant))
			SetTensorQuant(mat, "_Scale", quant);
		if(transposeWeight)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		ctx.Blit(output, mat);
		return output;
	}
	Texture _Reduce(Texture input, Keyword func, int groups=1, Vector4? indexRange=default, Texture rangeOffset=default, Matrix4x4? linear=default, int indexMod=0) {
		Debug.Assert(ctx.Size1(input) % groups == 0 || indexMod > 0);
		var groupSize = ctx.Size1(input) / groups;
		var lod = 0;
		if(func == Keyword.REDUCE_SUMPOW)
			lod = Mathf.Max(0, Mathf.FloorToInt(Mathf.Log(groupSize, 2)/2-1));
		else if(indexMod == 0 && groupSize >= reduceSplit) {
			var size1 = groups << Mathf.CeilToInt(Mathf.Log(groupSize, 2)/2);
			if(groups == 1 || ctx.Size1(input) % size1 == 0) { // disallow padding if groups > 1
				var input2 = _Reduce(input, func, groups:size1, indexRange:indexRange, rangeOffset:rangeOffset, indexMod:groupSize);
				var output2 = _Reduce(input2, func, groups:groups, linear:linear, indexMod:-1);
				ctx.Release(input2);
				return output2;
			}
		}
		var output = ctx.GPUTensor(ctx.Size0(input), groups, dtype:VertexAttributeFormat.Float32, lod:lod);
		var mat = ctx.Operator(kernels["Reduce"]);
		EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(indexMod > 0)
			mat.SetInt("_IndexMod", indexMod);
		else if(indexMod < 0)
			EnableOption(mat, Keyword.INPUT_REDUCED);
		if(indexRange != default)
			mat.SetVector("_IndexRange", indexRange.Value);
		if(rangeOffset)
			SetTensor(mat, "_Offset", rangeOffset);
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
	Texture _Normalize(Texture input, Keyword func, Keyword reduceFunc, int groups=1, float scale=1f, Vector4? indexRange=default, Texture rangeOffset=default) {
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
		mat.SetFloat("_Scale", scale);
		ctx.Blit(output, mat);
		ctx.Release(reduce);
		return output;
	}
	public Texture Fusion(Texture input, float scale=1f, Texture mul=default, Texture add=default, Keyword func=default, Vector4? indexRange=default, Texture rangeOffset=default, Vector4 defaultValue=default) {
		Debug.Assert(!mul || (ctx.Size0(input) % ctx.Size0(mul) == 0 && ctx.Size1(input) % ctx.Size1(mul) == 0));
		Debug.Assert(!add || (ctx.Size0(input) % ctx.Size0(add) == 0 && ctx.Size1(input) % ctx.Size1(add) == 0));
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dataType);
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(indexRange != default)
			mat.SetVector("_IndexRange", indexRange.Value);
		if(rangeOffset)
			SetTensor(mat, "_Offset", rangeOffset);
		mat.SetVector("_Default", defaultValue);
		mat.SetVector("_Weight", scale * Vector4.one);
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
		SetTensor(mat, "_Output", output, size);
		SetTensor(mat, "_Input",  input, size);
		mat.SetVector("_OutputOff", new Vector4(outputOffset.x, outputOffset.y, 0, 0));
		mat.SetVector("_InputOff",  new Vector4(inputOffset.x, inputOffset.y, 0, 0));
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Scatter(RenderTexture output, Texture index, Texture src, int chan=0) {
		// NOTE: this is actually fake: it only uses index[0,0]
		Debug.Assert(ctx.Size0(index) == ctx.Size0(src));
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output, ctx.Size(src));
		SetTensor(mat, "_Input",  src);
		SetTensor(mat, "_Offset", index);
		mat.SetVector("_OutputOff",  new Vector4(0, 0, chan==0?1:0, chan==1?1:0));
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Rotary(Texture input, Texture rotary, int groups=1) {
		Debug.Assert(ctx.Size1(input) % groups == 0 && ctx.Size1(input)/groups % 2 == 0 && ctx.Size1(rotary) % 2 == 0);
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
	public Texture Softmax(Texture input, int groups=1, float scale=1f, Vector4? indexRange=default, Texture rangeOffset=default) {
		var temp = _Normalize(input, Keyword.FUNC_SOFTMAX_LINF, Keyword.REDUCE_MINMAX, groups:groups, scale:scale, indexRange:indexRange, rangeOffset:rangeOffset);
		var output = _Normalize(temp, Keyword.FUNC_NORMALIZE_L1, Keyword.REDUCE_SUMPOW, groups:groups);
		ctx.Release(temp);
		return output;
	}
	public Texture Gumbel(Texture input, float temperature) {
		return Fusion(input, func:Keyword.FUNC_GUMBEL, scale:temperature, add:input);
	}

	void SetTensor(Material mat, string name, Texture tex) {
		mat.SetTexture($"{name}Tex", tex);
		mat.SetVector($"{name}Dim",  new Vector4(ctx.Size0(tex), ctx.Size1(tex), ctx.Tile1(tex), ctx.Lod(tex)));
	}
	void SetTensor(Material mat, string name, Texture tex, Vector2Int size) {
		mat.SetTexture($"{name}Tex", tex);
		mat.SetVector($"{name}Dim",  new Vector4(size.x, size.y, ctx.Tile1(tex), ctx.Lod(tex)));
	}
	void SetTensorQuant(Material mat, string name, Texture tex) {
		SetTensor(mat, name, tex);
		EnableOption(mat, UnityEngine.Experimental.Rendering.GraphicsFormatUtility.IsUNormFormat(tex.graphicsFormat)
			? Keyword.WEIGHT_QUANTIZED_E8 : Keyword.WEIGHT_QUANTIZED_S24_Z8);
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
		WEIGHT_QUANTIZED_S24_Z8,
		WEIGHT_QUANTIZED_E8,
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