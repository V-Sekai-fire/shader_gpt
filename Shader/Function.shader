Shader "GPT/Function" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_ReduceDim("_ReduceDim", Vector) = (1, 1, 0, 0)
	_OffsetDim("_OffsetDim", Vector) = (0, 0, 0, 0)
	_MulDim   ("_MulDim",    Vector) = (0, 0, 0, 0)
	_AddDim   ("_AddDim",    Vector) = (0, 0, 0, 0)
	_RotaryDim("_RotaryDim", Vector) = (0, 0, 0, 0)
	[HideInInspector]_OutputTex("_OutputTex", 2D) = "black" {}
	[NoScaleOffset]  _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset]  _ReduceTex("_ReduceTex", 2D) = "black" {}
	[NoScaleOffset]  _OffsetTex("_OffsetTex", 2D) = "black" {}
	[NoScaleOffset]  _MulTex   ("_MulTex",    2D) = "black" {}
	[NoScaleOffset]  _AddTex   ("_AddTex",    2D) = "black" {}
	[NoScaleOffset]  _RotaryTex("_RotaryTex", 2D) = "black" {}
	_OutputOff("_OutputOff", Vector) = (0, 0, 1, 1)
	_InputOff ("_InputOff",  Vector) = (0, 0, 1, 1)
	_Window   ("_Window",    Vector) = (0, 1048576, 0, 0)
	_Default  ("_Default",   Vector) = (0, 0, 0, 0)
	_Scale    ("_Scale", Float) = 1
	_Eps      ("_Eps",   Float) = 0
	_Mul      ("_Mul",   Vector) = (1, 1, 1, 1)
	_Add      ("_Add",   Vector) = (0, 0, 0, 0)
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
Texture2D<float4> _InputTex;  uint4 _InputDim;
Texture2D<float4> _ReduceTex; uint4 _ReduceDim;
Texture2D<float4> _OffsetTex; uint4 _OffsetDim;
Texture2D<float4> _MulTex;    uint4 _MulDim;
Texture2D<float4> _AddTex;    uint4 _AddDim;
Texture2D<float4> _RotaryTex; uint4 _RotaryDim;
uniform int2 _OutputOff;
uniform int4 _InputOff;
uniform float4 _Window;
uniform float4 _Default;
uniform float _Eps;
uniform float _Scale;
uniform float4 _Mul;
uniform float4 _Add;

float4 main(uint2 pos) {
	// output[i,j] = act(reduce(input[i,j]) * weight + bias)

	float4 X = LOAD_TENSOR(_Input, _InputOff.xy+_InputOff.zw*pos);
	float4 R = LOAD_TENSOR(_Reduce, pos.xy*_ReduceDim.xy/_InputDim.xy);
	float4 O = X;
	int4 index = pos.y%(_InputDim.y/_ReduceDim.y)*4 + uint4(0,1,2,3);
	int2 range = _Window.xy + dot(_Window.zw, LOAD_TENSOR(_Offset, uint2(pos.x, 0)).xy);
	bool4 mask = range.x <= index && index < range.y;
	#if defined(FUNC_GROUPNORM)
		O = (X*R[0]-R[1]) * rsqrt(_Eps*(R[0]*R[0]) + max(0, R[2]*R[0]-R[1]*R[1])); // R is sum of powers
	#elif defined(FUNC_NORMALIZE_L1)
		O = X / R[1]; // R is sum of powers
	#elif defined(FUNC_SOFTMAX_LINF)
		O = saturate(exp((X - R.y) * _Scale)); // R is minmax
	#elif defined(FUNC_GUMBEL)
		uint4 seed = uint4(pos.xy, asuint(_Time.y), asuint(_SinTime.w));
		O = -log(-log((pcg4d(seed)>>9u)/8388608.0 + 0.5/8388608.0)); // be careful to avoid input 0,1
	#elif defined(FUNC_ROTARY)
		uint j = pos.y%(_InputDim.y/_ReduceDim.y);
		uint dim = _RotaryDim.y/2;
		if(j < dim) {
			float4 reX = X;
			float4 imX = LOAD_TENSOR(_Input, _InputOff.xy+_InputOff.zw*(pos+uint2(0,dim)));
			float4 reY = LOAD_TENSOR(_Rotary, uint2(pos.x, j));
			float4 imY = LOAD_TENSOR(_Rotary, uint2(pos.x, j+dim));
			O = reX*reY - imX*imY; // real part
		} else if(j < dim*2) {
			float4 imX = X;
			float4 reX = LOAD_TENSOR(_Input, _InputOff.xy+_InputOff.zw*(pos-uint2(0,dim)));
			float4 imY = LOAD_TENSOR(_Rotary, uint2(pos.x, j));
			float4 reY = LOAD_TENSOR(_Rotary, uint2(pos.x, j-dim));
			O = reX*imY + imX*reY; // imaginary part
		}
	#endif

	O = mask ? O : _Default;
	O *= _Mul;
	if(_MulDim.x)
		O *= LOAD_TENSOR(_Mul, pos.xy*_MulDim.xy/_InputDim.xy);
	O += _Add;
	if(_AddDim.x)
		O += LOAD_TENSOR(_Add, pos.xy*_AddDim.xy/_InputDim.xy);

	#if defined(FUNC_GELU)
		O = gelu(O);
	#elif defined(FUNC_GELU_NEW)
		O = gelu_tanh(O);
	#elif defined(FUNC_RELU)
		O = max(0,O) + _Eps * min(0,O); // leaky relu
	#elif defined(FUNC_SIGMOID)
		O = sigmoid(O);
	#elif defined(FUNC_SILU)
		O = silu(O);
	#elif defined(FUNC_TANH)
		O = tanh(O);
	#endif
	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = getThreadId(screenPos, _OutputDim);
	pos -= _OutputOff;
	if(!all(0 <= int2(pos) && int2(pos) < int2(_OutputDim.xy)))
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
#pragma shader_feature _ FUNC_GROUPNORM FUNC_SOFTMAX_LINF FUNC_NORMALIZE_L1 FUNC_GUMBEL FUNC_ROTARY\
	FUNC_GELU FUNC_GELU_NEW FUNC_RELU FUNC_SIGMOID FUNC_SILU FUNC_TANH
ENDHLSL
	}
}
}