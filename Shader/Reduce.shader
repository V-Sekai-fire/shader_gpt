Shader "GPT/Reduce" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
	[HideInInspector]_OutputTex("_OutputTex",2D) = "black" {}
	[NoScaleOffset]  _InputTex ("_InputTex", 2D) = "black" {}
	_RangeMask("_RangeMask", Vector) = (0, 0, 0, 65536) // maxTextureSize*4
	_Linear0("_Linear0", Vector) = (1, 0, 0, 0)
	_Linear1("_Linear1", Vector) = (0, 1, 0, 0)
	_Linear2("_Linear2", Vector) = (0, 0, 1, 0)
	_Linear3("_Linear3", Vector) = (0, 0, 0, 1)
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
Texture2D<float4> _InputTex; uint4 _InputDim;
uniform int4  _RangeMask;
uniform float4 _Linear0;
uniform float4 _Linear1;
uniform float4 _Linear2;
uniform float4 _Linear3;

static const float oo = 3e38;

float4 main(uint2 pos) {
	// torch.mean, torch.aminmax
	// output[i,j].x = torch.mean(pow(input[i,j*K+k][c], 1), dim=(k,c))
	// output[i,j].y = torch.mean(pow(input[i,j*K+k][c], 2), dim=(k,c))
	// output[i,j].xz = torch.min(min4(input[i,j*K+k][c], dim=(k,c))
	// output[i,j].yw = torch.max(min4(input[i,j*K+k][c], dim=(k,c))

	uint K = _InputDim.y/_OutputDim.y, jK = pos.y*K;
	float4 O0 = 0;
	float4 O1 = 0;
	float4 O2 = 0;
	float4 O3 = 0;
#if defined(REDUCE_MINMAX)
	O0 = +oo;
	O1 = -oo;
#endif
	int2 range = pos.x * _RangeMask.xy + _RangeMask.zw;
	for(uint k=0; k<K; k++) {
		float4 X = _InputTex.mips[_InputDim.w][uint2(pos.x,jK+k).yx];
		int4 index = k*4 + uint4(0,1,2,3);
		bool4 mask = range.x <= index && index < range.y;
		// TODO: impl mask for all reductions 
		#if defined(REDUCE_MINMAX)
			O2 = mask && X < O0 ? k : O2;
			O3 = mask && X > O1 ? k : O3;
			O0 = mask ? min(X, O0) : O0;
			O1 = mask ? max(X, O1) : O1;
		#elif defined(REDUCE_MOMENT)
			O0 += X;
			O1 += X*X;
		#endif
	}
#if defined(REDUCE_MINMAX)
	float4 O = float4(+oo, -oo, 0, 0);
	[unroll] for(uint c=0; c<4; c++) {
		O[2] = O0[c] < O[0] ? O2[c]*4+c : O[2];
		O[3] = O1[c] > O[1] ? O3[c]*4+c : O[3];
		O[0] = min(O0[c], O[0]);
		O[1] = max(O1[c], O[1]);
	}
#elif defined(REDUCE_MOMENT)
	float4 O = float4(dot(0.25/K, O0), dot(0.25/K, O1), 0, 0);
#endif
	// return 
	// 	+(_Linear0 == 0 ? 0 : _Linear0*O[0])
	// 	+(_Linear1 == 0 ? 0 : _Linear1*O[1])
	// 	+(_Linear2 == 0 ? 0 : _Linear2*O[2])
	// 	+(_Linear3 == 0 ? 0 : _Linear3*O[3]);
	return _Linear0*O[0] + _Linear1*O[1] + _Linear2*O[2] + _Linear3*O[3];
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = floor(screenPos.yx);
	// NOTE: test shows calling getGroupThreadIdAndSize is expensive for reduce
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
#pragma shader_feature REDUCE_MOMENT REDUCE_MINMAX
ENDHLSL
	}
}
}