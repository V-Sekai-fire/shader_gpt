using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public class TensorNN {
	public TensorContext ctx;
	public Dictionary<string, Shader> kernels;
	public Dictionary<Texture, Texture> quants = new Dictionary<Texture, Texture>();
	public int linearLod = 2; // 3 has higher occupancy but similar gpu time
	public int reduceSplit = 64;

	public Texture IndexSelect(Texture input, (Texture tex, int chan) index, bool inputT=false, bool axis1=false) {
		var output = axis1
			? ctx.GPUTensor(inputT ? (ctx.Size1(input)+3)/4 : ctx.Size0(input), index.tex ? ctx.Size1(index.tex) : index.chan)
			: ctx.GPUTensor(index.tex ? ctx.Size0(index.tex) : index.chan, inputT ? (ctx.Size0(input)+3)/4 : ctx.Size1(input));
		var mat = ctx.Operator(kernels["Gather"]);
		SetTensor(mat, "_Output", output);
		if(axis1)
			EnableOption(mat, Keyword.AXIS_LAST);
		if(input)
			SetTensor(mat, "_Input", input);
		if(quants.TryGetValue(input, out var quant))
			SetTensorQuant(mat, "_Quant", quant);
		if(inputT)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		if(index.tex)
			SetTensor(mat, "_Index", index.tex);
		mat.SetInt("_IndexChan", index.chan);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture IndexCopy(RenderTexture output, (Texture tex, int chan) index, Texture input, float fill=0, bool axis1=false) {
		const int batchSize = 4096;
		Debug.Assert(!input || (axis1 ? ctx.Size1(index.tex) == ctx.Size1(input) : ctx.Size0(index.tex) == ctx.Size0(input)));
		var count = axis1 ? ctx.Size1(index.tex)*4 : ctx.Size0(index.tex);
		for(int off=0; off<count; off+=batchSize)
		for(int mask=axis1?1:15; mask<16; mask<<=1) {
			var mat = ctx.Operator(kernels["Scatter"]);
			SetTensor(mat, "_Output", output);
			if(axis1)
				EnableOption(mat, Keyword.AXIS_LAST);
			if(input)
				SetTensor(mat, "_Input", input);
			mat.SetVector("_Input", fill * Vector4.one);
			if(index.tex)
				SetTensor(mat, "_Index", index.tex);
			mat.SetVector("_IndexOff", axis1 ? new Vector2(0,off) : new Vector2(off,0));
			mat.SetInt("_IndexChan", index.chan);
			mat.SetInt("_ColorMask", mask);
			ctx.Blit(output, mat);
		}
		return output;
	}
	public Texture Linear(Texture input, Texture weight, Texture bias=null, bool weightT=false, int heads=1, int weightHeads=0) {
		if(weightHeads == 0)
			weightHeads = heads;
		Debug.Assert(ctx.Size1(input)%heads == 0 && ctx.Size1(weight)%weightHeads == 0 && ctx.Size0(weight)%4 == 0 && heads%weightHeads == 0);
		Debug.Assert(ctx.Size1(input)/heads == (weightT ? ctx.Size0(weight)/4 : ctx.Size1(weight)/weightHeads));
		var size1 = (weightT ? ctx.Size1(weight)/weightHeads : ctx.Size0(weight)/4) * heads;
		var lazyMips = ctx.Lod(input) == 0 && ctx.Lod(weight) == 0; // when adjacent operator is independent
		var output = ctx.GPUTensor(ctx.Size0(input), size1, dtype:ctx.DType(input), lod:linearLod, autoGenMips:!lazyMips);
		var mat = ctx.Operator(kernels["Linear"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(quants.TryGetValue(weight, out var quant))
			SetTensorQuant(mat, "_Quant", quant);
		if(weightT)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		if(bias)
			SetTensor(mat, "_Bias", bias);
		ctx.Blit(output, mat);
		return output;
	}
	Texture _Reduce(Texture input, Keyword func, int groups=1,
			Vector4? window=null, Texture offset=null, Matrix4x4? linear=null, int indexMod=0) {
		Debug.Assert(ctx.Size1(input) % groups == 0 || indexMod > 0);
		var groupSize = ctx.Size1(input) / groups;
		var lod = 0;
		if(func == Keyword.REDUCE_SUMPOW)
			lod = Mathf.Max(0, Mathf.FloorToInt(Mathf.Log(groupSize, 2)/2-1));
		else if(indexMod == 0 && groupSize >= reduceSplit) {
			var size1 = groups << Mathf.CeilToInt(Mathf.Log(groupSize, 2)/2);
			if(groups == 1 || ctx.Size1(input) % size1 == 0) { // disallow padding if groups > 1
				var input2 = _Reduce(input, func, groups:size1, window:window, offset:offset, indexMod:groupSize);
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
		if(window != default)
			mat.SetVector("_Window", window.Value);
		if(offset)
			SetTensor(mat, "_Offset", offset);
		if(linear != default) {
			mat.SetVector("_Linear0", linear.Value.GetColumn(0));
			mat.SetVector("_Linear1", linear.Value.GetColumn(1));
			mat.SetVector("_Linear2", linear.Value.GetColumn(2));
			mat.SetVector("_Linear3", linear.Value.GetColumn(3));
		}
		ctx.Blit(output, mat);
		return output;
	}
	Texture _Normalize(Texture input, Keyword func, Keyword reduceFunc, int groups=1,
			Texture mul=null, Texture add=null, float eps=0f, float scale=1f,
			Vector4? window=null, Texture offset=null, Matrix4x4? linear=null) {
		var reduce = _Reduce(input, reduceFunc, groups:groups, window:window, offset:offset, linear:linear);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Reduce", reduce);
		if(window != default)
			mat.SetVector("_Window", window.Value);
		if(offset)
			SetTensor(mat, "_Offset", offset);
		if(mul)
			SetTensor(mat, "_Mul", mul);
		if(add)
			SetTensor(mat, "_Add", add);
		mat.SetFloat("_Eps", eps);
		mat.SetFloat("_Scale", scale);
		ctx.Blit(output, mat);
		ctx.Release(reduce);
		return output;
	}
	public Texture Fusion(Texture input, float scale=1f, Texture mul=null, Texture add=null, Keyword func=0, float eps=0,
			Vector4? window=null, Texture offset=null, Vector4 @default=default) {
		Debug.Assert(!mul || (ctx.Size0(input) % ctx.Size0(mul) == 0 && ctx.Size1(input) % ctx.Size1(mul) == 0));
		Debug.Assert(!add || (ctx.Size0(input) % ctx.Size0(add) == 0 && ctx.Size1(input) % ctx.Size1(add) == 0));
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(window != default)
			mat.SetVector("_Window", window.Value);
		if(offset)
			SetTensor(mat, "_Offset", offset);
		mat.SetVector("_Default", @default);
		if(mul)
			SetTensor(mat, "_Mul", mul);
		if(add)
			SetTensor(mat, "_Add", add);
		mat.SetFloat("_Eps", eps);
		mat.SetVector("_Mul", scale * Vector4.one);
		if(func != default)
			EnableOption(mat, func);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Copy(RenderTexture output, Texture input, Vector2Int size,
			Vector2Int outputOffset=default, Vector2Int inputOffset=default) {
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
	public Texture Rotary(Texture input, Texture rotary, int groups=1) {
		Debug.Assert(ctx.Size1(input) % groups == 0 && ctx.Size1(input)/groups % 2 == 0 && ctx.Size1(rotary) % 2 == 0);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_ROTARY);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Rotary", rotary);
		mat.SetVector("_ReduceDim", new Vector2(ctx.Size0(input), groups));
		ctx.Blit(output, mat);
		return output;
	}

	public Texture Transpose(Texture input, int size0) {
		return IndexSelect(input, (null, size0), inputT:true);
	}
	public Texture ArgMax(Texture input, Vector4? window=null) {
		return _Reduce(input, Keyword.REDUCE_MINMAX, window:window,
			linear:new Matrix4x4(default, default, default, new Vector4(1,0,0,0)));
	}
	public Texture GroupNorm(Texture input, Texture weight, Texture bias, float eps, int groups=1, bool rmsNorm=false) {
		Debug.Assert(ctx.Size0(weight) == 1 && ctx.Size1(weight) == ctx.Size1(input));
		Debug.Assert(rmsNorm ? !bias : (ctx.Size0(bias) == 1 && ctx.Size1(bias) == ctx.Size1(input)));
		return _Normalize(input, Keyword.FUNC_GROUPNORM, reduceFunc:Keyword.REDUCE_SUMPOW, groups:groups,
			mul:weight, add:bias, eps:eps, linear:Matrix4x4.Scale(new Vector4(1, rmsNorm?0:1, 1, 1)));
	}
	public Texture Softmax(Texture input, int groups=1, float scale=1f, Vector4? window=null, Texture offset=null) {
		var temp = _Normalize(input, Keyword.FUNC_SOFTMAX_LINF, Keyword.REDUCE_MINMAX, groups:groups, scale:scale, window:window, offset:offset);
		var output = _Normalize(temp, Keyword.FUNC_NORMALIZE_L1, Keyword.REDUCE_SUMPOW, groups:groups);
		ctx.Release(temp);
		return output;
	}
	public Texture Gumbel(Texture input, float scale) {
		return Fusion(input, func:Keyword.FUNC_GUMBEL, scale:scale, add:input);
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
		AXIS_LAST,
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
		FUNC_RELU,
		FUNC_GELU,
		FUNC_GELU_NEW,
		FUNC_SILU,
	}
}
}