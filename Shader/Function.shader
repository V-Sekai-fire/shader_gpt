Shader "GPT/Function" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
	_ReduceDim("_ReduceDim", Vector) = (1, 1, 1, 0)
	_WeightDim("_WeightDim", Vector) = (0, 0, 0, 0)
	_BiasDim  ("_BiasDim",   Vector) = (0, 0, 0, 0)
	_RotaryDim("_RotaryDim", Vector) = (0, 0, 0, 0)
	[HideInInspector]_OutputTex("_OutputTex",2D) = "black" {}
	[NoScaleOffset] _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset] _ReduceTex("_ReduceTex", 2D) = "black" {}
	[NoScaleOffset] _OffsetTex("_OffsetTex", 2D) = "black" {}
	[NoScaleOffset] _WeightTex("_WeightTex", 2D) = "black" {}
	[NoScaleOffset] _BiasTex  ("_BiasTex",   2D) = "black" {}
	[NoScaleOffset] _RotaryTex("_RotaryTex", 2D) = "black" {}
	_InputOff ("_InputOff",  Vector) = (0, 0, 0, 0)
	_OutputOff("_OutputOff", Vector) = (0, 0, 0, 0)
	_IndexRange("_IndexRange", Vector) = (0, 65536, 0, 0) // maxTextureSize*4
	_Eps("_Eps", Float) = 0
	_Weight("_Weight", Vector) = (1, 1, 1, 1)
	_Bias  ("_Bias",   Vector) = (0, 0, 0, 0)
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint2 _OutputDim;
Texture2D<float4> _InputTex;  uint4 _InputDim;
Texture2D<float4> _ReduceTex; uint2 _ReduceDim;
Texture2D<float4> _OffsetTex;
Texture2D<float4> _WeightTex; uint4 _WeightDim;
Texture2D<float4> _BiasTex;   uint4 _BiasDim;
Texture2D<float4> _RotaryTex; uint2 _RotaryDim;
uniform uint2 _InputOff;
uniform uint4 _OutputOff;
uniform float4 _IndexRange;
uniform float _Eps;
uniform float4 _Weight;
uniform float4 _Bias;

float4 main(uint2 pos) {
	// output[i,j] = func(input[i,j])

	float4 X = _InputTex.mips[_InputDim.w][uint2(_InputOff.xy+pos.xy).yx];
	float4 R = _ReduceTex[uint2(pos.xy*_ReduceDim.xy/_InputDim.xy).yx];
	float4 O = X;
	int4 index = pos.y%(_InputDim.y/_ReduceDim.y)*4 + uint4(0,1,2,3);
	int2 range = _IndexRange.xy + dot(_IndexRange.zw, _OffsetTex[uint2(pos.x,0).yx].xy);
	bool4 mask = range.x <= index && index < range.y;
	#if defined(FUNC_GROUPNORM)
		O = mask ? (X*R[0]-R[1]) * rsqrt(_Eps*(R[0]*R[0]) + max(0, R[2]*R[0]-R[1]*R[1])) : 0; // R is sum of powers
	#elif defined(FUNC_NORMALIZE_L1)
		O = mask ? X / R[1] : 0; // R is sum of powers
	#elif defined(FUNC_SOFTMAX_LINF)
		O = mask ? saturate(exp(X - R.y)) : 0; // R is minmax
	#elif defined(FUNC_GUMBEL)
		uint4 seed = uint4(pos.xy, asuint(_Time.y), asuint(_SinTime.w));
		O = -log(-log((pcg4d(seed)>>9u)/8388608.0 + 0.5/8388608.0)); // be careful to avoid input 0,1
	#elif defined(FUNC_ROTARY)
		uint j = pos.y%(_InputDim.y/_ReduceDim.y);
		uint dim = _RotaryDim.y/2;
		if(j < dim) {
			float4 reX = X;
			float4 imX = _InputTex.mips[_InputDim.w][uint2(_InputOff.xy+uint2(pos.x,pos.y+dim)).yx];
			float4 reY = _RotaryTex[uint2(pos.x,j).yx];
			float4 imY = _RotaryTex[uint2(pos.x,j+dim).yx];
			O = reX*reY - imX*imY; // real part
		} else if(j < dim*2) {
			float4 imX = X;
			float4 reX = _InputTex.mips[_InputDim.w][uint2(_InputOff.xy+uint2(pos.x,pos.y-dim)).yx];
			float4 imY = _RotaryTex[uint2(pos.x,j).yx];
			float4 reY = _RotaryTex[uint2(pos.x,j-dim).yx];
			O = reX*imY + imX*reY; // imaginary part
		}
	#endif

	O *= _Weight;
	if(_WeightDim.x)
		O *= _WeightTex.mips[_WeightDim.w][uint2(pos.xy*_WeightDim.xy/_InputDim.xy).yx];
	O += _Bias;
	if(_BiasDim.x)
		O += _BiasTex.mips[_BiasDim.w][uint2(pos.xy*_BiasDim.xy/_InputDim.xy).yx];

	#if defined(FUNC_GELU)
		O = gelu(O);
	#elif defined(FUNC_GELU_NEW)
		O = gelu_new(O);
	#elif defined(FUNC_SILU)
		O = silu(O);
	#endif
	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = floor(screenPos.yx);
	int2 off = _OutputOff.xy + int2(dot(_OutputOff.zw, _OffsetTex[uint2(0,0).yx].xy), 0);
	if(!all(off <= pos && pos < off + _OutputDim.xy))
		discard;
	return main(pos-off);
}
ENDHLSL
	Pass {
		Cull Off
HLSLPROGRAM
#pragma target 5.0
#pragma vertex vertQuad
#pragma fragment frag
#pragma shader_feature _ FUNC_GROUPNORM FUNC_SOFTMAX_LINF FUNC_NORMALIZE_L1 FUNC_GELU FUNC_GELU_NEW FUNC_SILU FUNC_GUMBEL FUNC_ROTARY
ENDHLSL
	}
}
}