Shader "GPT/Linear" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
	[HideInInspector]_OutputTex("_OutputTex",2D) = "black" {}
	[NoScaleOffset] _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset] _WeightTex("_WeightTex", 2D) = "black" {}
	_Head("_Head", Int) = 0
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
Texture2D<float4> _InputTex; uint4 _InputDim;
Texture2D<float4> _WeightTex;
uniform uint _Head;

float4 main(uint2 pos, uint threadId, uint groupSize) {
	// torch.nn.Linear with multi-head support
	// output[i,h*J+j][jj] += input[i,h*K+k][kk] * (transpose ? weight[k*4+kk,h*J+j][jj] : weight[j*4+jj,h*K+k][kk])

	uint J = _OutputDim.y/_Head, j = pos.y%J;
	uint K = _InputDim.y/_Head, hK = pos.y/J*K;
	float4 O = 0;
	#ifdef TRANSPOSE_WEIGHT
		for(uint k=threadId; k<K; k+=groupSize) {
			float4 X = _InputTex.mips[_InputDim.w][uint2(pos.x,hK+k).yx];
			O += X[0] * _WeightTex[uint2(k*4+0,pos.y).yx];
			O += X[1] * _WeightTex[uint2(k*4+1,pos.y).yx];
			O += X[2] * _WeightTex[uint2(k*4+2,pos.y).yx];
			O += X[3] * _WeightTex[uint2(k*4+3,pos.y).yx];
		}
	#else
		for(uint k=threadId; k<K; k+=groupSize) {
			float4 X = _InputTex.mips[_InputDim.w][uint2(pos.x,hK+k).yx];
			O[0] += dot(X, _WeightTex[uint2(j*4+0,hK+k).yx]);
			O[1] += dot(X, _WeightTex[uint2(j*4+1,hK+k).yx]);
			O[2] += dot(X, _WeightTex[uint2(j*4+2,hK+k).yx]);
			O[3] += dot(X, _WeightTex[uint2(j*4+3,hK+k).yx]);
		}
	#endif
	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = floor(screenPos.yx);
	uint2 group = getGroupThreadIdAndSize(pos, _OutputDim.w);
	if(any(pos >= _OutputDim.xy))
		discard;
	return main(pos, group.x, group.y) * group.y;
}
ENDHLSL
	Pass {
		Cull Off
HLSLPROGRAM
#pragma target 5.0
#pragma vertex vertQuad
#pragma fragment frag
#pragma shader_feature TRANSPOSE_WEIGHT
ENDHLSL
	}
}
}