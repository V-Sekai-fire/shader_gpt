Shader "GPT/Embedding" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
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

uint2 _OutputDim;
Texture2D<float2> _InputTex; uint4 _InputDim;
Texture2D<float4> _Weight0Tex;
Texture2D<float4> _Weight1Tex;
Texture2D<float4> _Scale0Tex; uint4 _Scale0Dim;
Texture2D<float4> _Scale1Tex; uint4 _Scale1Dim;

float4 main(uint2 pos) {
	// torch.nn.functional.embedding_bag(mode="sum")
	// weight[i,j] *= scale[i/4,j][i%4]
	// output[i,j][jj] += transpose ? weight_k[j*4+jj,input[i,0][k]/4][input[i,0][k]%4] : weight_k[input[i,0][k],j][jj]

	uint2 X = round(_InputTex.mips[_InputDim.w][uint2(pos.x,0).yx]);
	float4 O;
#ifdef TRANSPOSE_WEIGHT
	// tested: error rate of per-channel block q8 is smaller than per-word
	float4 offset0, scale0 = dequantizeScale(_Scale0Tex[uint2(pos.y,X[0]/4).yx], offset0, _Scale0Dim.x != 0);
	float4 offset1, scale1 = dequantizeScale(_Scale1Tex[uint2(pos.y,X[1]/4).yx], offset1, _Scale1Dim.x != 0);
	[unroll] for(int c=0; c<4; c++) {
		O[c]  = dequantizeWeight(_Weight0Tex[uint2(pos.y*4+c,X[0]/4).yx], offset0[c])[X[0]%4] * scale0[c];
		O[c] += dequantizeWeight(_Weight1Tex[uint2(pos.y*4+c,X[1]/4).yx], offset1[c])[X[1]%4] * scale1[c];
	}
#else
	float4 offset0, scale0 = dequantizeScale(_Scale0Tex[uint2(X[0]/4,pos.y).yx], offset0, _Scale0Dim.x != 0);
	float4 offset1, scale1 = dequantizeScale(_Scale1Tex[uint2(X[1]/4,pos.y).yx], offset1, _Scale1Dim.x != 0);
	O  = dequantizeWeight(_Weight0Tex[uint2(X[0],pos.y).yx], offset0[X[0]%4]) * scale0[X[0]%4];
	O += dequantizeWeight(_Weight1Tex[uint2(X[1],pos.y).yx], offset1[X[1]%4]) * scale1[X[1]%4];
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