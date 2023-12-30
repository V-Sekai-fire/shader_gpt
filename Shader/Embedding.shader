Shader "GPT/Embedding" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
	[HideInInspector]_OutputTex("_OutputTex", 2D) = "black" {}
	[NoScaleOffset] _InputTex  ("_InputTex",  2D) = "black" {}
	[NoScaleOffset] _WeightTex ("_WeightTex", 2D) = "black" {}
	[NoScaleOffset] _Weight2Tex("_Weight2Tex",2D) = "black" {}
	[NoScaleOffset] _ScaleTex  ("_ScaleTex",  2D) = "white" {}
	[NoScaleOffset] _Scale2Tex ("_Scale2Tex", 2D) = "white" {}
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint2 _OutputDim;
Texture2D<float2> _InputTex; uint4 _InputDim;
Texture2D<float4> _WeightTex;
Texture2D<float4> _Weight2Tex;
Texture2D<float4> _ScaleTex;
Texture2D<float4> _Scale2Tex;

float4 main(uint2 pos) {
	// torch.nn.Embedding
	// weight[i,j] *= scale[i/4,j][i%4]
	// output[i,j][jj] += transpose ? weight_k[j*4+jj,input[i,0][k]/4][input[i,0][k]%4] : weight_k[input[i,0][k],j][jj]

	uint2 X = round(_InputTex.mips[_InputDim.w][uint2(pos.x,0).yx]);
	float4 O;
#ifdef TRANSPOSE_WEIGHT
	// tested: error rate of per-channel block q8 is smaller than per-word
	float4 scale  = dequantizeScale(_ScaleTex [uint2(pos.y,X[0]/4).yx]);
	float4 scale2 = dequantizeScale(_Scale2Tex[uint2(pos.y,X[1]/4).yx]);
	[unroll] for(int c=0; c<4; c++) {
		O[c]  = dequantizeWeight(_WeightTex [uint2(pos.y*4+c,X[0]/4).yx])[X[0]%4] * scale [c];
		O[c] += dequantizeWeight(_Weight2Tex[uint2(pos.y*4+c,X[1]/4).yx])[X[1]%4] * scale2[c];
	}
#else
	float4 scale  = dequantizeScale(_ScaleTex [uint2(X[0]/4,pos.y).yx]);
	float4 scale2 = dequantizeScale(_Scale2Tex[uint2(X[1]/4,pos.y).yx]);
	O  = dequantizeWeight(_WeightTex [uint2(X[0],pos.y).yx]) * scale [X[0]%4];
	O += dequantizeWeight(_Weight2Tex[uint2(X[1],pos.y).yx]) * scale2[X[1]%4];
#endif
	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = floor(screenPos.yx);
	if(any(pos >= _OutputDim))
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
#pragma shader_feature TRANSPOSE_WEIGHT
#pragma shader_feature QUANTIZE_WEIGHT
ENDHLSL
	}
}
}