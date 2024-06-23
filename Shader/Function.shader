Shader "GPT/Function" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_ReduceDim("_ReduceDim", Vector) = (1, 1, 0, 0)
	_WindowDim("_WindowDim", Vector) = (0, 0, 0, 0)
	_MulDim   ("_MulDim",    Vector) = (0, 0, 0, 0)
	_AddDim   ("_AddDim",    Vector) = (0, 0, 0, 0)
	_RotaryDim("_RotaryDim", Vector) = (0, 0, 0, 0)
	[HideInInspector]
	_OutputTex("_OutputTex", 2D) = "black" {}
	_InputTex ("_InputTex",  2D) = "black" {}
	_ReduceTex("_ReduceTex", 2D) = "black" {}
	_WindowTex("_WindowTex", 2D) = "black" {}
	_MulTex   ("_MulTex",    2D) = "black" {}
	_AddTex   ("_AddTex",    2D) = "black" {}
	_RotaryTex("_RotaryTex", 2D) = "black" {}
	_Window   ("_Window",    Vector) = (0, 1000000000, 0, 0)
	_Default  ("_Default",   Vector) = (0, 0, 0, 0)
	_Scale    ("_Scale", Float) = 1
	_Eps      ("_Eps",   Float) = 0
	_Mul      ("_Mul",   Vector) = (1, 1, 1, 1)
	_Add      ("_Add",   Vector) = (0, 0, 0, 0)
	_FoldSize ("_FoldSize", Vector) = (0, 0, 0, 0)
	_FoldOff  ("_FoldOff", Int) = 0
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim; int4 _OutputTex_ST;
DEFINE_TEXTURE2D(_InputTex);  uint4 _InputDim;
DEFINE_TEXTURE2D(_ReduceTex); uint4 _ReduceDim; // y = group count for non-reduce functions
DEFINE_TEXTURE2D(_WindowTex); uint4 _WindowDim;
DEFINE_TEXTURE2D(_MulTex);    uint4 _MulDim;
DEFINE_TEXTURE2D(_AddTex);    uint4 _AddDim;
DEFINE_TEXTURE2D(_RotaryTex); uint4 _RotaryDim;
uniform float4 _Window;
uniform float4 _Default;
uniform float _Eps;
uniform float _Scale;
uniform float4 _Mul;
uniform float4 _Add;
uniform uint4 _FoldSize;
uniform int _FoldOff;

float4 main(uint2 pos) {
	// output[i,j] = func(input[i,j]) * mul + add

	float4 X = LOAD_TENSOR(_Input, _InputTex_ST.xy*pos);
	float4 R = LOAD_TENSOR(_Reduce, pos.xy/(_OutputDim.xy/_ReduceDim.xy));
	float4 O = X;
	int4 index = pos.y%(_OutputDim.y/_ReduceDim.y)*4 + uint4(0,1,2,3);
	int2 range = _Window.xy + dot(_Window.zw, LOAD_TENSOR(_Window, uint2(min(_WindowDim.x-1, pos.x), 0)).xy);
	bool4 mask = range.x <= index && index < range.y;
	#if defined(FUNC_GROUPNORM) // torch.nn.functional.group_norm
		O = (X*R[0]-R[1]) * rsqrt(_Eps*(R[0]*R[0]) + max(0, R[2]*R[0]-R[1]*R[1])); // R[n] is sum of n-th powers
	#elif defined(FUNC_SOFTMAX) // torch.nn.functional.softmax
		O = saturate(exp(X*_Scale-R[0])/R[1]); // exp(R[0])*R[1] is sum of exps

	#elif defined(FUNC_GELU) // torch.nn.functional.gelu
		O = gelu(X);
	#elif defined(FUNC_GELU_NEW) // torch.nn.functional.gelu(approximate="tanh")
		O = gelu_tanh(X);
	#elif defined(FUNC_RELU) // torch.nn.functional.leaky_relu
		O = max(0,X) + _Eps * min(0,X);
	#elif defined(FUNC_SIGMOID) // torch.nn.functional.sigmoid
		O = sigmoid(X);
	#elif defined(FUNC_SILU) // torch.nn.functional.silu
		O = silu(X);
	#elif defined(FUNC_TANH) // torch.nn.functional.tanh
		O = tanh(X);

	#elif defined(FUNC_EXP) // torch.exp
		O = exp(X*_Scale);
	#elif defined(FUNC_GUMBEL) // torch.distributions.gumbel
		uint4 seed = uint4(pos.xy, asuint(_Time.y), asuint(_SinTime.w));
		O = -log(-log((pcg4d(seed)>>9u)/8388608.0 + 0.5/8388608.0)); // be careful to avoid input 0,1
	#elif defined(FUNC_NORMAL) // torch.distributions.normal
		uint4 seed = uint4(pos.xy, asuint(_Time.y), asuint(_SinTime.w));
		uint4 rand = pcg4d(seed);
		float2 radius = sqrt(-2*log((rand.xy>>9u)/8388608.0 + 0.5/8388608.0));
		float2 angle  = int2(rand.zw-2147483648) / 2147483648.0 * UNITY_PI;
		O = float4(cos(angle), sin(angle)) * radius.xyxy;

	#elif defined(FUNC_ROTARY) // transformers.models.llama.apply_rotary_pos_emb
		uint j = pos.y%(_OutputDim.y/_ReduceDim.y);
		uint J = _RotaryDim.y;
		if(j < J) {
			uint2 k = j+(J+uint2(0,1))/2;
			float4 Z, W, Y = LOAD_TENSOR(_Rotary, uint2(pos.x, j));
			if(J%2 == 0) {
				W = LOAD_TENSOR(_Rotary, uint2(pos.x, k.x%J));
				Z = LOAD_TENSOR(_Input, _InputTex_ST.xy*uint2(pos.x, pos.y-j+k.x%J));
			} else {
				W.xy = LOAD_TENSOR(_Rotary, uint2(pos.x, k.x%J)).zw;
				W.zw = LOAD_TENSOR(_Rotary, uint2(pos.x, k.y%J)).xy;
				Z.xy = LOAD_TENSOR(_Input, _InputTex_ST.xy*uint2(pos.x, pos.y-j+k.x%J)).zw;
				Z.zw = LOAD_TENSOR(_Input, _InputTex_ST.xy*uint2(pos.x, pos.y-j+k.y%J)).xy;
			}
			O = k.xxyy >= J ? Z*Y + X*W : X*Y - Z*W;
		}

	#elif defined(FUNC_NARROW) // torch.narrow
		uint J = _InputDim.y/_ReduceDim.y;
		uint k = pos.y/(_OutputDim.y/_ReduceDim.y);
		index += range.x;
		mask = index < range.y;
		[unroll] for(uint c=0; c<4; c++) {
			uint j = index[c];
			O[c] = j/4 < J ? LOAD_TENSOR(_Input, _InputTex_ST.xy*uint2(pos.x, k*J+j/4))[j%4] : 0;
		}
	#elif defined(FUNC_RESHAPE) // torch.reshape
		uint idx = pos.x * _OutputDim.y + pos.y;
		O = LOAD_TENSOR(_Input, _InputTex_ST.xy*uint2(idx/_InputDim.y, idx%_InputDim.y));
	#elif defined(FUNC_UNFOLD) // torch.nn.functional.unfold
		uint kernel_size = _FoldSize.x, dilation = _FoldSize.y;
		uint stride = _FoldSize.z,      dilation_stride = _FoldSize.w;
		[unroll] for(uint c=0; c<4; c++) {
			uint j = pos.y*4+c;
			uint r = j % kernel_size;
			uint q = j / kernel_size % dilation;
			uint p = j / kernel_size / dilation;
			uint k = (p*dilation_stride + r)*dilation + q*stride + uint(_FoldOff);
			O[c] = k/4 < _InputDim.y ? LOAD_TENSOR(_Input, _InputTex_ST.xy*uint2(pos.x, k/4))[k%4] : 0;
		}
	#endif

	O = mask ? O : _Default;

	float4 mul4 = _Mul;
	float4 add4 = _Add;
	if(_MulDim.x)
		mul4 *= LOAD_TENSOR(_Mul, pos.xy%_MulDim.xy);
	if(_AddDim.x)
		add4 += LOAD_TENSOR(_Add, pos.xy%_AddDim.xy);
	O = mad(_Mul == 0 ? 0 : O, mul4, add4); // clear infinity properly if zero

	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = getThreadId(screenPos, _OutputDim);
	pos -= _OutputTex_ST.zw;
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
	FUNC_ROTARY\
	FUNC_NARROW FUNC_RESHAPE FUNC_UNFOLD
ENDHLSL
	}
}
}