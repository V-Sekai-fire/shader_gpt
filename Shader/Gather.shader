Shader "GPT/Gather" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_IndexDim ("_IndexDim",  Vector) = (0, 0, 0, 0) // zero = enumeration
	_QuantDim ("_QuantDim",  Vector) = (0, 0, 0, 0)
	[HideInInspector]_OutputTex("_OutputTex", 2D) = "black" {}
	[NoScaleOffset]  _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset]  _IndexTex ("_IndexTex",  2D) = "black" {}
	[NoScaleOffset]  _QuantTex ("_QuantTex",  2D) = "black" {}
	_IndexChan("_IndexChan", Int) = 0
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
Texture2D<float4> _InputTex; uint4 _InputDim;
Texture2D<float4> _IndexTex; uint4 _IndexDim;
Texture2D<float4> _QuantTex; uint4 _QuantDim;
uniform uint _IndexChan;

float4 main(uint2 pos) {
	// torch.index_select(input, axis, index)
	// output[i,j][jj] = axis == 1 ? input[i,index[c,j][jj]/4][index[c,j][jj]%4]
	// 	: transpose ? input[j*4+jj,index[i,c]/4][index[i,c]%4] : input[index[i,c],j][jj]

	uint S = _InputDim.y / _QuantDim.y;
	float4 O;
#ifndef AXIS_LAST
	uint idx = _IndexDim.x ? LOAD_TENSOR(_Index, uint2(pos.x, _IndexChan/4))[_IndexChan%4] : pos.x;
#ifdef WEIGHT_TRANSPOSED
	float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(pos.y, idx/4/S)), offset);
	[unroll] for(int c=0; c<4; c++)
		O[c] = dequantizeWeight(LOAD_TENSOR(_Input, uint2(pos.y*4+c, idx/4)), offset[c])[idx%4] * scale[c];
	O = idx/4 < _InputDim.y ? O : 0;
#else
	float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(idx/4, pos.y/S)), offset);
	O = dequantizeWeight(LOAD_TENSOR(_Input, uint2(idx, pos.y)), offset[idx%4]) * scale[idx%4];
	O = idx < _InputDim.x ? O : 0;
#endif
#else
	uint4 idx4 = LOAD_TENSOR(_Index, uint2(_IndexChan, pos.y));
	[unroll] for(int c=0; c<4; c++) {
		uint idx = idx4[c];
		float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(pos.x/4, idx/4/S)), offset);
		O[c] = dequantizeWeight(LOAD_TENSOR(_Input, uint2(pos.x, idx/4)), offset[pos.x%4])[idx%4] * scale[pos.x%4];
	}
	O = idx4/4 < _InputDim.y ? O : 0;
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
#pragma shader_feature _ WEIGHT_TRANSPOSED AXIS_LAST
#pragma shader_feature _ WEIGHT_QUANTIZED_S24_Z8 WEIGHT_QUANTIZED_E8
ENDHLSL
	}
}
}