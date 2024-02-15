Shader "GPT/Embedding" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
	_Weight0Dim("_Weight0Dim", Vector) = (0, 0, 0, 0)
	_Weight1Dim("_Weight1Dim", Vector) = (0, 0, 0, 0)
	_Scale0Dim("_Scale0Dim", Vector) = (0, 0, 0, 0)
	_Scale1Dim("_Scale1Dim", Vector) = (0, 0, 0, 0)
	[HideInInspector]_OutputTex("_OutputTex", 2D) = "black" {}
	[NoScaleOffset] _InputTex  ("_InputTex",  2D) = "black" {}
	[NoScaleOffset] _Weight0Tex("_Weight0Tex",2D) = "black" {}
	[NoScaleOffset] _Weight1Tex("_Weight1Tex",2D) = "black" {}
	[NoScaleOffset] _Scale0Tex ("_Scale0Tex", 2D) = "black" {}
	[NoScaleOffset] _Scale1Tex ("_Scale1Tex", 2D) = "black" {}
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
Texture2D<float4> _InputTex; uint4 _InputDim;
Texture2D<float4> _Weight0Tex; uint4 _Weight0Dim;
Texture2D<float4> _Weight1Tex; uint4 _Weight1Dim;
Texture2D<float4> _Scale0Tex; uint4 _Scale0Dim;
Texture2D<float4> _Scale1Tex; uint4 _Scale1Dim;

float4 main(uint2 pos) {
	// torch.nn.functional.embedding_bag(mode="sum")
	// weight[i,j] *= scale[i/4,j][i%4]
	// output[i,j][jj] += transpose ? weight_k[j*4+jj,input[i,0][k]/4][input[i,0][k]%4] : weight_k[input[i,0][k],j][jj]

	uint2 X = round(loadTensor(_InputTex, pos.x, 0, _InputDim.w).xy);
	float4 O;
#ifdef WEIGHT_TRANSPOSED
	// NOTE: wide tensor is only supported on transposed weight to reduce overhead
	// tested: error rate of per-channel block q8 is smaller than per-word
	float4 offset0, scale0 = dequantizeScale(loadTensor(_Scale0Tex, pos.y, X[0]/4, _Scale0Dim), offset0, _Scale0Dim.x != 0);
	float4 offset1, scale1 = dequantizeScale(loadTensor(_Scale1Tex, pos.y, X[1]/4, _Scale1Dim), offset1, _Scale1Dim.x != 0);
	[unroll] for(int c=0; c<4; c++) {
		O[c]  = dequantizeWeight(loadTensor(_Weight0Tex, pos.y*4+c, X[0]/4, _Weight0Dim), offset0[c])[X[0]%4] * scale0[c];
		O[c] += dequantizeWeight(loadTensor(_Weight1Tex, pos.y*4+c, X[1]/4, _Weight1Dim), offset1[c])[X[1]%4] * scale1[c];
	}
#else
	float4 offset0, scale0 = dequantizeScale(loadTensor(_Scale0Tex, X[0]/4, pos.y), offset0, _Scale0Dim.x != 0);
	float4 offset1, scale1 = dequantizeScale(loadTensor(_Scale1Tex, X[1]/4, pos.y), offset1, _Scale1Dim.x != 0);
	O  = dequantizeWeight(loadTensor(_Weight0Tex, X[0], pos.y), offset0[X[0]%4]) * scale0[X[0]%4];
	O += dequantizeWeight(loadTensor(_Weight1Tex, X[1], pos.y), offset1[X[1]%4]) * scale1[X[1]%4];
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
#pragma shader_feature WEIGHT_QUANTIZED
ENDHLSL
	}
}
}