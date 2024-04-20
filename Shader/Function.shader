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
	// output[i,j] = func(input[i,j]) * mul + add

	float4 X = LOAD_TENSOR(_Input, _InputOff.xy+_InputOff.zw*pos);
	float4 R = LOAD_TENSOR(_Reduce, pos.xy/(_OutputDim.xy/_ReduceDim.xy));
	float4 O = X;
	int4 index = pos.y%(_OutputDim.y/_ReduceDim.y)*4 + uint4(0,1,2,3);
	int2 range = _Window.xy + dot(_Window.zw, LOAD_TENSOR(_Offset, uint2(pos.x, 0)).xy);
	bool4 mask = range.x <= index && index < range.y;
	#if defined(FUNC_GROUPNORM)
		// torch.nn.functional.group_norm
		O = (X*R[0]-R[1]) * rsqrt(_Eps*(R[0]*R[0]) + max(0, R[2]*R[0]-R[1]*R[1])); // R[n] is sum of n-th powers
	#elif defined(FUNC_SOFTMAX)
		// torch.nn.functional.softmax
		O = saturate(exp(X*_Scale-R[0])/R[1]); // exp(R[0])*R[1] is sum of exps

	#elif defined(FUNC_GELU)
		// torch.nn.functional.gelu(approximate="none")
		O = gelu(X);
	#elif defined(FUNC_GELU_NEW)
		// torch.nn.functional.gelu(approximate="tanh")
		O = gelu_tanh(X);
	#elif defined(FUNC_RELU)
		// torch.nn.functional.leaky_relu
		O = max(0,X) + _Eps * min(0,X);
	#elif defined(FUNC_SIGMOID)
		// torch.nn.functional.sigmoid
		O = sigmoid(X);
	#elif defined(FUNC_SILU)
		// torch.nn.functional.silu
		O = silu(X);
	#elif defined(FUNC_TANH)
		// torch.nn.functional.tanh
		O = tanh(X);

	#elif defined(FUNC_EXP)
		// torch.exp
		O = exp(X*_Scale);
	#elif defined(FUNC_GUMBEL)
		// torch.distributions.gumbel.Gumbel
		uint4 seed = uint4(pos.xy, asuint(_Time.y), asuint(_SinTime.w));
		O = -log(-log((pcg4d(seed)>>9u)/8388608.0 + 0.5/8388608.0)); // be careful to avoid input 0,1
	#elif defined(FUNC_NORMAL)
		// torch.distributions.normal.Normal
		uint4 seed = uint4(pos.xy, asuint(_Time.y), asuint(_SinTime.w));
		uint4 rand = pcg4d(seed);
		float2 radius = sqrt(-2*log((rand.xy>>9u)/8388608.0 + 0.5/8388608.0));
		float2 angle  = int2(rand.zw-2147483648) / 2147483648.0 * UNITY_PI;
		O = float4(cos(angle), sin(angle)) * radius.xyxy;

	#elif defined(FUNC_ROTARY)
		uint j = pos.y%(_OutputDim.y/_ReduceDim.y);
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
		O *= LOAD_TENSOR(_Mul, pos.xy/(_OutputDim.xy/_MulDim.xy));
	O += _Add;
	if(_AddDim.x)
		O += LOAD_TENSOR(_Add, pos.xy/(_OutputDim.xy/_AddDim.xy));

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
#pragma shader_feature _\
	FUNC_GROUPNORM FUNC_SOFTMAX\
	FUNC_GELU FUNC_GELU_NEW FUNC_RELU FUNC_SIGMOID FUNC_SILU FUNC_TANH\
	FUNC_EXP FUNC_GUMBEL FUNC_NORMAL\
	FUNC_ROTARY
ENDHLSL
	}
}
}