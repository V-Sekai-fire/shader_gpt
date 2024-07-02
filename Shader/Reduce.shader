Shader "GPT/Reduce" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_WindowDim("_WindowDim", Vector) = (0, 0, 0, 0)
	[HideInInspector]
	_OutputTex("_OutputTex", 2D) = "black" {}
	_InputTex ("_InputTex",  2D) = "black" {}
	_WindowTex("_WindowTex", 2D) = "black" {}
	_Window  ("_Window",  Vector) = (0, 1000000000, 0, 0)
	_IndexMod("_IndexMod", Int) = 1
	_Scale   ("_Scale", Float) = 1
	_Linear0 ("_Linear0", Vector) = (1, 0, 0, 0)
	_Linear1 ("_Linear1", Vector) = (0, 1, 0, 0)
	_Linear2 ("_Linear2", Vector) = (0, 0, 1, 0)
	_Linear3 ("_Linear3", Vector) = (0, 0, 0, 1)
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
DEFINE_TEXTURE2D(_InputTex);  uint4 _InputDim;
DEFINE_TEXTURE2D(_WindowTex); uint4 _WindowDim;
uniform uint4  _Window;
uniform uint   _IndexMod;
uniform float  _Scale;
uniform float4 _Linear0;
uniform float4 _Linear1;
uniform float4 _Linear2;
uniform float4 _Linear3;

static const float oo = asfloat(0x7f7fffff);
float min4(float4 v) {
	return min(min(v.x, v.y), min(v.z, v.w));
}
float max4(float4 v) {
	return max(max(v.x, v.y), max(v.z, v.w));
}

float4 main(uint2 pos, uint threadId, uint groupSize) {
	// torch.sum, torch.aminmax
	// output[i,j][e] = torch.sum(pow(input[i,j*K+k][c], e), dim=(k,c))
	// output[i,j].xz = torch.min(input[i,j*K+k][c], dim=(k,c))
	// output[i,j].yw = torch.max(input[i,j*K+k][c], dim=(k,c))

	uint K = (_InputDim.y+_OutputDim.y-1)/_OutputDim.y, jK = pos.y*K;
	int2 range = _Window.xy + _Window.z * (uint)(int)LOAD_TENSOR(_Window, uint2(min(_WindowDim.x-1, pos.x), 0))[_Window.w];
	#if defined(REDUCE_SUMPOW)
		float4 O = 0;
	#elif defined(REDUCE_SUMEXP)
		float4 O = float4(-oo, 0, 0, 0);
	#elif defined(REDUCE_MINMAX)
		float4 O = float4(+oo, -oo, 0, 0);
	#endif
	K = min(K, uint(max(0, int(_InputDim.y-jK)))); // clamp out-of-bound read on _InputTex
	for(uint k=threadId; k<K; k+=groupSize) {
		float4 X = LOAD_TENSOR(_Input, uint2(pos.x, jK+k));
		#if !defined(INPUT_REDUCED)
			int4 index = (jK%_IndexMod+k)*4 + uint4(0,1,2,3);
			bool4 mask = range.x <= index && index < range.y;
			#if defined(REDUCE_SUMPOW)
				X = float4(dot(mask, 1), dot(mask, X), dot(mask, X*X), dot(mask, X*X*X));
			#elif defined(REDUCE_SUMEXP)
				float4 Xmax = mask ? X*_Scale : -oo;
				X[0] = max4(Xmax);
				X[1] = dot(1, saturate(exp(Xmax-X[0])));
			#elif defined(REDUCE_MINMAX)
				float4 Xmin = mask ? X : +oo;
				float4 Xmax = mask ? X : -oo;
				X[0] = min4(Xmin);
				X[1] = max4(Xmax);
				X[2] = min4(Xmin == X[0] ? index : +oo);
				X[3] = min4(Xmax == X[1] ? index : +oo);
			#endif
		#endif
		#if defined(REDUCE_SUMPOW)
			O += X;
		#elif defined(REDUCE_SUMEXP)
			float e = saturate(exp(-abs(X[0]-O[0])));
			O[1] = X[0] > O[0] ? X[1] + O[1]*e : X[1]*e + O[1];
			O[0] = max(X[0], O[0]);
		#elif defined(REDUCE_MINMAX)
			O[2] = X[0] < O[0] ? X[2] : O[2];
			O[3] = X[1] > O[1] ? X[3] : O[3];
			O[0] = min(X[0], O[0]);
			O[1] = max(X[1], O[1]);
		#endif
	}
	// return 
	// 	+(_Linear0 == 0 ? 0 : _Linear0*O[0])
	// 	+(_Linear1 == 0 ? 0 : _Linear1*O[1])
	// 	+(_Linear2 == 0 ? 0 : _Linear2*O[2])
	// 	+(_Linear3 == 0 ? 0 : _Linear3*O[3]);
	return mul(O, float4x4(_Linear0, _Linear1, _Linear2, _Linear3));
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	#if defined(REDUCE_SUMPOW)
		uint4 pos = getThreadIdAndGroupSize(screenPos, _OutputDim);
	#else
		uint4 pos = uint4(getThreadId(screenPos, _OutputDim), 0, 1);
	#endif
	if(any(pos.xy >= _OutputDim.xy))
		discard;
	return main(pos.xy, pos.z, pos.w) * pos.w;
}
ENDHLSL
	Pass {
		Cull Off
HLSLPROGRAM
#pragma target 5.0
#pragma vertex vertQuad
#pragma fragment frag
#pragma shader_feature INPUT_REDUCED
#pragma shader_feature REDUCE_SUMPOW REDUCE_SUMEXP REDUCE_MINMAX
ENDHLSL
	}
}
}