Shader "GPT/Linear" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
	_WeightDim("_WeightDim", Vector) = (1, 1, 1, 0)
	_ScaleDim ("_ScaleDim",  Vector) = (1, 1, 1, 0)
	[HideInInspector]_OutputTex("_OutputTex",2D) = "black" {}
	[NoScaleOffset] _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset] _WeightTex("_WeightTex", 2D) = "black" {}
	[NoScaleOffset] _ScaleTex ("_ScaleTex",  2D) = "black" {}
	_Scale("_Scale", Float) = 1
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
Texture2D<float4> _InputTex;  uint4 _InputDim;
Texture2D<float4> _WeightTex; uint4 _WeightDim;
Texture2D<float4> _ScaleTex;  uint4 _ScaleDim;
uniform float _Scale;

float4 main(uint2 pos, uint threadId, uint groupSize) {
	// torch.nn.functional.linear(bias=None) with multi-head support
	// weight[i,j] *= scale[i/4,j][i%4]
	// output[i,h*J+j][jj] += input[i,h*K+k][kk] * (transpose ? weight[k*4+kk,h/D*J+j][jj] : weight[j*4+jj,h/D*K+k][kk])

	#ifdef WEIGHT_TRANSPOSED
		uint H = _InputDim.y*4  / _WeightDim.x;
		uint D = _OutputDim.y   / _WeightDim.y;
	#else
		uint H = _OutputDim.y*4 / _WeightDim.x;
		uint D = _InputDim.y    / _WeightDim.y;
	#endif
	uint J = _OutputDim.y / H;
	uint K = _InputDim.y  / H;
	uint j = pos.y % J;
	uint h = pos.y / J;
	float4 O = 0;
	for(uint k=threadId; k<K; k+=groupSize) {
		float4 X = loadTensor(_InputTex, pos.x, h*K+k, _InputDim.w);
		#ifdef WEIGHT_TRANSPOSED
			// NOTE: wide tensor is only supported on transposed weight to reduce overhead
			float4 offset, scale = dequantizeScale(loadTensor(_ScaleTex, k, h/D*J+j, _ScaleDim), offset);
			O += mul(scale * X, float4x4(
				dequantizeWeight(loadTensor(_WeightTex, k*4+0, h/D*J+j, _WeightDim), offset[0]),
				dequantizeWeight(loadTensor(_WeightTex, k*4+1, h/D*J+j, _WeightDim), offset[1]),
				dequantizeWeight(loadTensor(_WeightTex, k*4+2, h/D*J+j, _WeightDim), offset[2]),
				dequantizeWeight(loadTensor(_WeightTex, k*4+3, h/D*J+j, _WeightDim), offset[3])));
		#else
			// tested: error rate of per-out-channel block q8 is 10%~50% smaller than per-input-channel
			// awq does this too: github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization/gemv_cuda.cu
			float4 offset, scale = dequantizeScale(loadTensor(_ScaleTex, j, h/D*K+k, _ScaleDim.w), offset);
			O += scale * mul(float4x4(
				dequantizeWeight(loadTensor(_WeightTex, j*4+0, h/D*K+k, _WeightDim.w), offset[0]),
				dequantizeWeight(loadTensor(_WeightTex, j*4+1, h/D*K+k, _WeightDim.w), offset[1]),
				dequantizeWeight(loadTensor(_WeightTex, j*4+2, h/D*K+k, _WeightDim.w), offset[2]),
				dequantizeWeight(loadTensor(_WeightTex, j*4+3, h/D*K+k, _WeightDim.w), offset[3])), X);
		#endif
	}
	O *= _Scale;
	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint4 pos = getThreadIdAndGroupSize(screenPos, _OutputDim);
	if(any(pos.xy >= _OutputDim.xy))
		discard;
	return main(pos.xy, pos.z, pos.w) * pos.w;
}
ENDHLSL
	Pass {
		Cull Off
HLSLPROGRAM
#pragma target 5.0
#pragma vertex vertQuad
#pragma fragment frag
#pragma shader_feature WEIGHT_TRANSPOSED
#pragma shader_feature WEIGHT_QUANTIZED
ENDHLSL
	}
}
}