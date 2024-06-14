using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System.Linq;

namespace ShaderGPT {
public class TensorNN {
	public TensorContext ctx;
	public Dictionary<string, Shader> kernels;
	public Dictionary<Texture, Texture> quantizers = new Dictionary<Texture, Texture>();
	public Dictionary<Texture, Texture> permuters = new Dictionary<Texture, Texture>();
	public int linearLod = 2; // 3 has higher occupancy but similar gpu time
	public int reduceSplit = 64;

	public Texture IndexSelect(TexView input, (TexView tex, int chan) index, bool inputT=false, bool axis1=false) {
		if(permuters.TryGetValue((Texture)input, out var permuter)) {
			Debug.Assert(!axis1); // only embedding is supported
			if(inputT)
				index = (IndexSelect(permuter, index, inputT:true), 1);
		}
		var size0 = !axis1 ? (index.tex ? ctx.Size0(index.tex) : index.chan) : (inputT ? (ctx.Size1(input)+3)/4 : ctx.Size0(input));
		var size1 =  axis1 ? (index.tex ? ctx.Size1(index.tex) : index.chan) : (inputT ? (ctx.Size0(input)+3)/4 : ctx.Size1(input));
		var output = ctx.GPUTensor(size0, size1,
			dtype:UnityEngine.Experimental.Rendering.GraphicsFormatUtility.IsUNormFormat(((Texture)input).graphicsFormat) ?
				(VertexAttributeFormat?)null : ctx.DType(input)); // avoid quantized type
		var mat = ctx.Operator(kernels["Gather"]);
		SetTensor(mat, "_Output", output);
		if(axis1)
			EnableOption(mat, Keyword.AXIS_LAST);
		if(input)
			SetTensor(mat, "_Input", input);
		if(quantizers.TryGetValue((Texture)input, out var quantizer))
			SetTensorQuant(mat, "_Quant", quantizer);
		if(inputT)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		if(index.tex)
			SetTensor(mat, "_Index", index.tex);
		mat.SetInt("_IndexChan", index.chan);
		ctx.Blit(output, mat);
		if(permuter) {
			if(inputT)
				ctx.Release((Texture)index.tex);
			else {
				var permOutput = IndexSelect(output, (permuter, 1), axis1:true);
				ctx.Release(output);
				return permOutput;
			}
		}
		return output;
	}
	public Texture IndexCopy(RenderTexture output, (TexView tex, int chan) index, TexView input, float value=0, bool axis1=false) {
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
				SetTensor(mat, "_Input", axis1
					? ctx.Slice(input, ctx.Size0(input), Mathf.Min(ctx.Size1(input)-off/4, batchSize/4), 0, off/4)
					: ctx.Slice(input, Mathf.Min(ctx.Size0(input)-off, batchSize), ctx.Size1(input), off, 0));
			else 
				mat.SetVector("_Input", value * Vector4.one);
			var intex = index.tex;
			if(intex)
				SetTensor(mat, "_Index", axis1
					? ctx.Slice(intex, ctx.Size0(intex), Mathf.Min(ctx.Size1(intex)-off/4, batchSize/4), 0, off/4)
					: ctx.Slice(intex, Mathf.Min(ctx.Size0(intex)-off, batchSize), ctx.Size1(intex), off, 0));
			mat.SetInt("_IndexChan", index.chan);
			mat.SetInt("_ColorMask", mask);
			ctx.Blit(output, mat);
		}
		return output;
	}
	public Texture Linear(TexView input, TexView weight, TexView bias=default, bool weightT=false, int heads=1, int weightHeads=0) {
		if(weightHeads == 0)
			weightHeads = heads;
		if(permuters.TryGetValue((Texture)weight, out var permuter)) {
			Debug.Assert(heads == 1 && weightHeads == 1);
			if(!weightT)
				input = IndexSelect(input, (permuter, 0), axis1:true);
		}
		var idim = !weightT ? ctx.Size1(weight)/weightHeads : (ctx.Size0(weight)+3)/4;
		var odim =  weightT ? ctx.Size1(weight)/weightHeads : (ctx.Size0(weight)+3)/4;
		Debug.Assert(ctx.Size1(input) == heads*idim && ctx.Size1(weight)%weightHeads == 0 && heads%weightHeads == 0);
		Debug.Assert(!bias || (ctx.Size0(bias) == 1 && ctx.Size1(bias) == heads*odim));
		var output = ctx.GPUTensor(ctx.Size0(input), heads*odim, dtype:ctx.DType(input), lod:-1,
			autoGenMips:(ctx.Lod(input)+ctx.Lod(weight) > 0)); // resolve mips if any input has mips
		var mat = ctx.Operator(kernels["Linear"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(quantizers.TryGetValue((Texture)weight, out var quantizer))
			SetTensorQuant(mat, "_Quant", quantizer);
		if(weightT)
			EnableOption(mat, Keyword.WEIGHT_TRANSPOSED);
		if(bias)
			SetTensor(mat, "_Bias", bias);
		ctx.Blit(output, mat);
		if(permuter) {
			if(!weightT)
				ctx.Release((Texture)input);
			else {
				var permOutput = IndexSelect(output, (permuter, 1), axis1:true);
				ctx.Release(output);
				return permOutput;
			}
		}
		return output;
	}
	Texture _Reduce(TexView input, Keyword func, int groups=1, float scale=1f,
			(Vector4,Texture)? window=null, Matrix4x4? linear=null, int indexMod=0) {
		Debug.Assert(ctx.Size1(input) % groups == 0 || indexMod > 0);
		var groupSize = ctx.Size1(input) / groups;
		var lod = 0;
		if(func == Keyword.REDUCE_SUMPOW)
			lod = Mathf.Max(0, Mathf.FloorToInt(Mathf.Log(groupSize, 2)/2-1));
		else if(indexMod == 0 && groupSize >= reduceSplit) {
			var size1 = groups << Mathf.CeilToInt(Mathf.Log(groupSize, 2)/2);
			if(groups == 1 || ctx.Size1(input) % size1 == 0) { // disallow padding if groups > 1
				var input2 = _Reduce(input, func, groups:size1, scale:scale, window:window, indexMod:groupSize);
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
		if(window != null) {
			mat.SetVector("_Window", window.Value.Item1);
			if(window.Value.Item2)
				SetTensor(mat, "_Window", window.Value.Item2);
		}
		if(linear != default) {
			mat.SetVector("_Linear0", linear.Value.GetColumn(0));
			mat.SetVector("_Linear1", linear.Value.GetColumn(1));
			mat.SetVector("_Linear2", linear.Value.GetColumn(2));
			mat.SetVector("_Linear3", linear.Value.GetColumn(3));
		}
		mat.SetFloat("_Scale", scale);
		ctx.Blit(output, mat);
		return output;
	}
	Texture _Normalize(TexView input, Keyword func, Keyword reduceFunc, int groups=1,
			TexView mul=default, TexView add=default, float eps=0f, float scale=1f,
			(Vector4,Texture)? window=null, Matrix4x4? linear=null) {
		var reduce = _Reduce(input, reduceFunc, groups:groups, scale:scale, window:window, linear:linear);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, func);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Reduce", reduce);
		if(window != null) {
			mat.SetVector("_Window", window.Value.Item1);
			if(window.Value.Item2)
				SetTensor(mat, "_Window", window.Value.Item2);
		}
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
	public Texture Fusion(TexView input, float scale=1f, float bias=0f, TexView mul=default, TexView add=default, Keyword func=0, float eps=0,
			(Vector4,Texture)? window=null, Vector4 @default=default, VertexAttributeFormat? dtype=null) {
		Debug.Assert(!mul || (ctx.Size0(input) % ctx.Size0(mul) == 0 && ctx.Size1(input) % ctx.Size1(mul) == 0));
		Debug.Assert(!add || (ctx.Size0(input) % ctx.Size0(add) == 0 && ctx.Size1(input) % ctx.Size1(add) == 0));
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:dtype??ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		if(window != null) {
			mat.SetVector("_Window", window.Value.Item1);
			if(window.Value.Item2)
				SetTensor(mat, "_Window", window.Value.Item2);
		}
		mat.SetVector("_Default", @default);
		if(mul)
			SetTensor(mat, "_Mul", mul);
		if(add)
			SetTensor(mat, "_Add", add);
		mat.SetVector("_Mul", scale * Vector4.one);
		mat.SetVector("_Add", bias * Vector4.one);
		mat.SetFloat("_Eps", eps);
		if(func != default)
			EnableOption(mat, func);
		ctx.Blit(output, mat);
		return output;
	}
	public TexView Copy(TexView output, TexView input, bool reshape=false) {
		Debug.Assert(reshape ? ctx.Size0(output)*ctx.Size1(output) == ctx.Size0(input)*ctx.Size1(input)
			: ctx.Size(output) == ctx.Size(input));
		var rt = (RenderTexture)output;
		var mat = ctx.Operator(kernels["Function"]);
		if(reshape)
			EnableOption(mat, Keyword.FUNC_RESHAPE);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		ctx.Blit(rt, mat);
		return output;
	}
	public Texture Flip(TexView input) {
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		mat.SetTextureOffset("_InputTex", new Vector2(ctx.Size0(input)-1+ctx.Offset0(input), ctx.Offset1(input)));
		mat.SetTextureScale ("_InputTex", new Vector2(-1, 1));
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Rotary(TexView input, TexView rotary, int groups=1) {
		Debug.Assert(ctx.Size1(input) % groups == 0);
		var output = ctx.GPUTensor(ctx.Size0(input), ctx.Size1(input), dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_ROTARY);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Rotary", rotary);
		mat.SetVector("_ReduceDim", new Vector2(1, groups));
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Narrow(TexView input, (Vector4,Texture) window, int groups=1) {
		Debug.Assert(ctx.Size1(input) % groups == 0);
		var size1 = ((int)(window.Item1.y-window.Item1.x)+3)/4 * groups;
		var output = ctx.GPUTensor(ctx.Size0(input), size1, dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_NARROW);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		mat.SetVector("_ReduceDim", new Vector2(1, groups));
		mat.SetVector("_Window", window.Item1);
		if(window.Item2)
			SetTensor(mat, "_Window", window.Item2);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Unfold(TexView input, int size1, int kernel_size, int dilation=1, int padding=0, int stride=1, int? dilation_stride=null) {
		if(kernel_size%4 == 0 && size1%(kernel_size/4) == 0) {
			var n = size1/(kernel_size/4);
			var lvl = Mathf.Max(0, Mathf.CeilToInt(Mathf.Log(size1, 2)-14));
			size1 = (kernel_size/4) * ((n + (1<<lvl)-1) &~ ((1<<lvl)-1)); // make sure wide tensor is aligned
		}
		var output = ctx.GPUTensor(ctx.Size0(input), size1, dtype:ctx.DType(input));
		var mat = ctx.Operator(kernels["Function"]);
		EnableOption(mat, Keyword.FUNC_UNFOLD);
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		mat.SetVector("_FoldSize", new Vector4(kernel_size, dilation, stride, dilation_stride??stride));
		mat.SetInt("_FoldOff", -padding);
		ctx.Blit((RenderTexture)output, mat);
		return output;
	}
	Texture Conv1d_unfolded(TexView input, TexView weight, TexView bias, int kernel_size, int stride) {
		var idim = (kernel_size+3*stride+3)/4;
		var wdim = (kernel_size+3)/4;
		Debug.Assert(ctx.Size1(input)%idim == 0);
		Debug.Assert(ctx.Size1(weight) == (kernel_size==1 ? (ctx.Size0(input)+3)/4 : ctx.Size0(input)*wdim));
		Debug.Assert(!bias || (ctx.Size0(bias) == 1 && ctx.Size1(bias) == (ctx.Size0(weight)+3)/4));
		var output = ctx.GPUTensor(ctx.Size0(weight), ctx.Size1(input)/idim, dtype:ctx.DType(input), lod:-1);
		var mat = ctx.Operator(kernels["Conv1d"]);
		EnableOption(mat, (Keyword)System.Enum.Parse(typeof(Keyword), $"CONV_K{kernel_size}_S{stride}"));
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(bias)
			SetTensor(mat, "_Bias", bias);
		ctx.Blit(output, mat);
		return output;
	}
	Texture ConvTranspose1d_unfolded(TexView input, TexView weight, TexView bias, int kernel_size, int stride) {
		const int idim = 1;
		var wdim = (kernel_size+3)/4;
		Debug.Assert(ctx.Size1(input)%idim == 0);
		Debug.Assert(ctx.Size0(weight) == ctx.Size0(input) && ctx.Size1(weight)%wdim == 0);
		Debug.Assert(!bias || (ctx.Size0(bias) == 1 && ctx.Size1(bias) == (ctx.Size1(weight)/wdim+3)/4));
		var output = ctx.GPUTensor(ctx.Size1(weight)/wdim, ctx.Size1(input)*(stride/2), dtype:ctx.DType(input), lod:-1);
		var mat = ctx.Operator(kernels["Conv1d"]);
		EnableOption(mat, (Keyword)System.Enum.Parse(typeof(Keyword), $"CONV_TRANSPOSE_K{kernel_size}_S{stride}"));
		SetTensor(mat, "_Output", output);
		SetTensor(mat, "_Input",  input);
		SetTensor(mat, "_Weight", weight);
		if(bias)
			SetTensor(mat, "_Bias", bias);
		ctx.Blit(output, mat);
		return output;
	}
	public Texture Conv1d(TexView input, TexView weight, TexView bias=default, int kernel_size=0, int dilation=1) {
		const int stride = 1;
		if(kernel_size == 1)
			return Conv1d_unfolded(input, weight, bias, kernel_size:kernel_size, stride:stride);
		var idim = (kernel_size+3*stride+3)/4;
		var input_uf = Unfold(input, size1:(ctx.Size1(input)+dilation-1)/dilation*dilation*idim,
			kernel_size:idim*4, dilation:dilation, dilation_stride:4, padding:(kernel_size-stride)/2*dilation);
		var output_uf = Conv1d_unfolded(input_uf, weight, bias, kernel_size:kernel_size, stride:stride);
		ctx.Release(input_uf);
		if(dilation == 1)
			return output_uf;
		var output = Unfold(output_uf, size1:ctx.Size1(input), kernel_size:dilation, dilation:4, dilation_stride:dilation); // fold
		ctx.Release(output_uf);
		return output;
	}
	public Texture ConvTranspose1d(TexView input, TexView weight, TexView bias=default, int kernel_size=0, int stride=1) {
		var input_uf = Unfold(input, size1:ctx.Size1(input)*2, kernel_size:4, dilation_stride:2, padding:1);
		var output_uf = ConvTranspose1d_unfolded(input_uf, weight, bias, kernel_size:kernel_size, stride:stride);
		ctx.Release(input_uf);
		return output_uf;
	}

	public Texture Transpose(TexView input, int size0) {
		return IndexSelect(input, (null, size0), inputT:true);
	}
	public Texture Sum(TexView input, (Vector4,Texture)? window=null) {
		return _Reduce(input, Keyword.REDUCE_SUMPOW, window:window,
			linear:new Matrix4x4(default, new Vector4(1,0,0,0), default, default));
	}
	public Texture ArgMax(TexView input, (Vector4,Texture)? window=null) {
		return _Reduce(input, Keyword.REDUCE_MINMAX, window:window,
			linear:new Matrix4x4(default, default, default, new Vector4(1,0,0,0)));
	}
	public Texture GroupNorm(TexView input, TexView weight, TexView bias, float eps, int groups=1, bool rms=false) {
		Debug.Assert(ctx.Size0(weight) == 1 && ctx.Size1(weight)*groups % ctx.Size1(input) == 0); // layernorm or groupnorm
		Debug.Assert(rms ? !bias : (ctx.Size0(bias) == 1 && ctx.Size1(bias) == ctx.Size1(weight)));
		return _Normalize(input, Keyword.FUNC_GROUPNORM, reduceFunc:Keyword.REDUCE_SUMPOW, groups:groups,
			mul:weight, add:bias, eps:eps, linear:Matrix4x4.Scale(new Vector4(1, rms?0:1, 1, 1)));
	}
	public Texture Softmax(TexView input, int groups=1, float scale=1f, (Vector4,Texture)? window=null) {
		return _Normalize(input, Keyword.FUNC_SOFTMAX, Keyword.REDUCE_SUMEXP, groups:groups, scale:scale, window:window);
	}
	public Texture Gumbel(TexView input, float scale) {
		return Fusion(input, func:Keyword.FUNC_GUMBEL, scale:scale, add:input);
	}

	void SetTensor(Material mat, string name, TexView view) {
		Debug.Assert(view);
		mat.SetTexture($"{name}Tex", (Texture)view);
		mat.SetVector($"{name}Dim",  new Vector4(ctx.Size0(view), ctx.Size1(view), ctx.Tile1(view), ctx.Lod(view)));
		mat.SetTextureOffset($"{name}Tex", new Vector2(ctx.Offset0(view), ctx.Offset1(view)));
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
		if(actFnRemap.TryGetValue(name, out var name2))
			name = name2;
		return (Keyword)System.Enum.Parse(typeof(Keyword), $"FUNC_{name.ToUpperInvariant()}");
	}
	static readonly Dictionary<string,string> actFnRemap = new Dictionary<string,string>() {
		{"swish", "silu"},
		{"gelu_pytorch_tanh", "gelu_new"},
	};

	public enum Keyword {
		None = 0,
		AXIS_LAST,
		WEIGHT_TRANSPOSED,
		WEIGHT_QUANTIZED_S24_Z8,
		WEIGHT_QUANTIZED_E8,
		INPUT_REDUCED,
		REDUCE_SUMPOW,
		REDUCE_SUMEXP,
		REDUCE_MINMAX,

		FUNC_GROUPNORM,
		FUNC_SOFTMAX,
		FUNC_GELU,
		FUNC_GELU_NEW,
		FUNC_RELU,
		FUNC_SIGMOID,
		FUNC_SILU,
		FUNC_TANH,

		FUNC_EXP,
		FUNC_GUMBEL,
		FUNC_NORMAL,
		FUNC_ROTARY,

		FUNC_NARROW,
		FUNC_RESHAPE,
		FUNC_UNFOLD,

		CONV_K1_S1, CONV_K3_S1, CONV_K5_S1, CONV_K7_S1, CONV_K11_S1,
		CONV_TRANSPOSE_K4_S2, CONV_TRANSPOSE_K16_S8,
	}
}
}