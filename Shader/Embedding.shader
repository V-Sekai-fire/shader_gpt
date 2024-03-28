Shader "GPT/Embedding" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (0, 0, 0, 0) // zero = enumeration
	_WeightDim("_WeightDim", Vector) = (1, 1, 0, 0)
	_ScaleDim ("_ScaleDim",  Vector) = (0, 0, 0, 0)
	[HideInInspector]_OutputTex("_OutputTex", 2D) = "black" {}
	[NoScaleOffset]  _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset]  _WeightTex("_WeightTex", 2D) = "black" {}
	[NoScaleOffset]  _ScaleTex ("_ScaleTex",  2D) = "black" {}
	_InputChan("_InputChan", Int) = 0
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
uniform uint _InputChan;

float4 main(uint2 pos) {
	// torch.nn.functional.embedding()
	// output[i,j][jj] = transpose ? weight[j*4+jj,input[i,c]/4][input[i,c]%4] : weight[input[i,c],j][jj]

	uint S = _WeightDim.y / _ScaleDim.y;

	uint idx = _InputDim.x ? LOAD_TENSOR(_Input, uint2(pos.x, _InputChan/4))[_InputChan%4] : pos.x;
	float4 O;
#ifdef WEIGHT_TRANSPOSED
	// tested: error rate of per-channel block q8 is smaller than per-word
	float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Scale, uint2(pos.y, idx/4/S)), offset);
	[unroll] for(int c=0; c<4; c++)
		O[c] = dequantizeWeight(LOAD_TENSOR(_Weight, uint2(pos.y*4+c, idx/4)), offset[c])[idx%4] * scale[c];
#else
	float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Scale, uint2(idx/4, pos.y/S)), offset);
	O = dequantizeWeight(LOAD_TENSOR(_Weight, uint2(idx, pos.y)), offset[idx%4]) * scale[idx%4];
#endif
	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = getThreadId(screenPos, _OutputDim);
	if(any(pos >= _OutputDim.xy))
		discard;
	return main(pos);
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