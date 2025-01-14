Shader "GPT/Linear" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_WeightDim("_WeightDim", Vector) = (1, 1, 0, 0)
	_QuantDim ("_QuantDim",  Vector) = (0, 0, 0, 0)
	_BiasDim  ("_BiasDim",   Vector) = (0, 0, 0, 0)
	[HideInInspector]
	_OutputTex("_OutputTex", 2D) = "black" {}
	_InputTex ("_InputTex",  2D) = "black" {}
	_WeightTex("_WeightTex", 2D) = "black" {}
	_QuantTex ("_QuantTex",  2D) = "black" {}
	_BiasTex  ("_BiasTex",   2D) = "black" {}
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
DEFINE_TEXTURE2D(_InputTex);  uint4 _InputDim;
DEFINE_TEXTURE2D(_WeightTex); uint4 _WeightDim;
DEFINE_TEXTURE2D(_QuantTex);  uint4 _QuantDim;
DEFINE_TEXTURE2D(_BiasTex);   uint4 _BiasDim;

float4 main(uint2 pos, uint threadId, uint groupSize) {
	// torch.nn.functional.linear with multi-head support
	// output == bias + torch.einsum("ihdk,khj->ihdj" if transpose else "ihdk,jhk->ihdj", input, weight)
	// output[i,h*J+j][jj] += input[i,h*K+k][kk] * (transpose ? weight[k*4+kk,h/D*J+j][jj] : weight[j*4+jj,h/D*K+k][kk])

	#ifdef WEIGHT_TRANSPOSED
		uint H = _InputDim.y  / ((_WeightDim.x+3)/4);
		uint D = _OutputDim.y / _WeightDim.y;
	#else
		uint H = _OutputDim.y / ((_WeightDim.x+3)/4);
		uint D = _InputDim.y  / _WeightDim.y;
	#endif
	uint J = _OutputDim.y / H;
	uint K = _InputDim.y  / H;
	uint S = _WeightDim.y / _QuantDim.y;
	uint j = pos.y % J;
	uint h = pos.y / J;
	float4 O = 0;
	float4 B = LOAD_TENSOR(_Bias, uint2(0, pos.y)); // load here for less divergence
	for(uint k=threadId; k<K; k+=groupSize) {
		float4 X = LOAD_TENSOR(_Input, uint2(pos.x, h*K+k));
		// for performance reason, when _WeightDim.x%4 != 0, out-of-bound read on _WeightTex is not handled
		#ifdef WEIGHT_TRANSPOSED
			float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(k, (h/D*J+j)/S)), offset);
			O += mul(scale * X, float4x4(
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(k*4+0, h/D*J+j)), offset[0]),
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(k*4+1, h/D*J+j)), offset[1]),
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(k*4+2, h/D*J+j)), offset[2]),
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(k*4+3, h/D*J+j)), offset[3])));
		#else
			float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(j, (h/D*K+k)/S)), offset);
			O += scale * mul(float4x4(
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(j*4+0, h/D*K+k)), offset[0]),
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(j*4+1, h/D*K+k)), offset[1]),
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(j*4+2, h/D*K+k)), offset[2]),
				dequantizeWeight(LOAD_TENSOR(_Weight, uint2(j*4+3, h/D*K+k)), offset[3])), X);
		#endif
	}
	O += B / groupSize; // accum here for less error
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
#pragma shader_feature _ WEIGHT_QUANTIZED_S24_Z8 WEIGHT_QUANTIZED_E8
ENDHLSL
	}
}
}