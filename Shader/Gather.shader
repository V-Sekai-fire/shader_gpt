Shader "GPT/Gather" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_IndexDim ("_IndexDim",  Vector) = (0, 0, 0, 0) // x=0: use enumeration
	_QuantDim ("_QuantDim",  Vector) = (0, 0, 0, 0)
	[HideInInspector]
	_OutputTex("_OutputTex", 2D) = "black" {}
	_InputTex ("_InputTex",  2D) = "black" {}
	_IndexTex ("_IndexTex",  2D) = "black" {}
	_QuantTex ("_QuantTex",  2D) = "black" {}
	_IndexChan("_IndexChan", Int) = 0
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
DEFINE_TEXTURE2D(_InputTex); uint4 _InputDim;
DEFINE_TEXTURE2D(_IndexTex); uint4 _IndexDim;
DEFINE_TEXTURE2D(_QuantTex); uint4 _QuantDim;
uniform uint _IndexChan;

float4 main(uint2 pos) {
	// torch.index_select with a sliced index
	// output == torch.index_select(input, axis, torch.select(index, 1-axis, chan))
	// output[i,j][jj] = axis == 1 ?
	//    (transpose ? input[index[c,j][jj],i/4][i%4] : input[i,index[c,j][jj]/4][index[c,j][jj]%4])
	// 	: (transpose ? input[j*4+jj,index[i,c]/4][index[i,c]%4] : input[index[i,c],j][jj])

	uint S = _InputDim.y / _QuantDim.y;
	float4 O;
#ifdef AXIS_LAST
	uint4 idx4 = _IndexDim.x ? LOAD_TENSOR(_Index, uint2(_IndexChan, pos.y)) : pos.y*4+uint4(0,1,2,3);
	[unroll] for(int c=0; c<4; c++) {
#ifdef WEIGHT_TRANSPOSED
		uint j = pos.x, i = idx4[c];
#else
		uint i = pos.x, j = idx4[c];
#endif
		float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(i/4, j/4/S)), offset);
		O[c] = dequantizeWeight(LOAD_TENSOR(_Input, uint2(i, j/4)), offset[i%4])[j%4] * scale[i%4];
	}
	// handle out-of-bound index values
#ifdef WEIGHT_TRANSPOSED
	O = idx4   < _InputDim.x ? O : 0;
#else
	O = idx4/4 < _InputDim.y ? O : 0;
#endif
#else
	uint idx = _IndexDim.x ? LOAD_TENSOR(_Index, uint2(pos.x, _IndexChan/4))[_IndexChan%4] : pos.x;
#ifdef WEIGHT_TRANSPOSED
	float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(pos.y, idx/4/S)), offset);
	[unroll] for(int c=0; c<4; c++)
		O[c] = dequantizeWeight(LOAD_TENSOR(_Input, uint2(pos.y*4+c, idx/4)), offset[c])[idx%4] * scale[c];
	O = idx/4 < _InputDim.y ? O : 0;
#else
	float4 offset, scale = dequantizeScale(LOAD_TENSOR(_Quant, uint2(idx/4, pos.y/S)), offset);
	O = dequantizeWeight(LOAD_TENSOR(_Input, uint2(idx, pos.y)), offset[idx%4]) * scale[idx%4];
	O = idx   < _InputDim.x ? O : 0;
#endif
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
#pragma shader_feature _ AXIS_LAST
#pragma shader_feature _ WEIGHT_TRANSPOSED
#pragma shader_feature _ WEIGHT_QUANTIZED_S24_Z8 WEIGHT_QUANTIZED_E8
ENDHLSL
	}
}
}