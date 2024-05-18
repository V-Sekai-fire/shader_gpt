Shader "GPT/Conv1d" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_WeightDim("_WeightDim", Vector) = (1, 1, 0, 0)
	_BiasDim  ("_BiasDim",   Vector) = (0, 0, 0, 0)
	[HideInInspector]
	_OutputTex("_OutputTex", 2D) = "black" {}
	_InputTex ("_InputTex",  2D) = "black" {}
	_WeightTex("_WeightTex", 2D) = "black" {}
	_BiasTex  ("_BiasTex",   2D) = "black" {}
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

#if defined(CONV_K1_S1)
static const uint kernel_size = 1, stride = 1;
#elif defined(CONV_K3_S1)
static const uint kernel_size = 3, stride = 1;
#elif defined(CONV_K5_S1)
static const uint kernel_size = 5, stride = 1;
#elif defined(CONV_K7_S1)
static const uint kernel_size = 7, stride = 1;
#elif defined(CONV_K11_S1)
static const uint kernel_size = 11, stride = 1;
#elif defined(CONV_TRANSPOSE_K4_S2)
#define TRANSPOSE
static const uint kernel_size = 4, stride = 2;
#elif defined(CONV_TRANSPOSE_K16_S8)
#define TRANSPOSE
static const uint kernel_size = 16, stride = 8;
#endif

uint4 _OutputDim;
DEFINE_TEXTURE2D(_InputTex);  uint4 _InputDim;
DEFINE_TEXTURE2D(_WeightTex); uint4 _WeightDim;
DEFINE_TEXTURE2D(_BiasTex);   uint4 _BiasDim;

float4 main(uint2 pos, uint threadId, uint groupSize) {
	// torch.nn.functional.conv1d/conv_transpose1d for batched short inputs
	float4 O = 0;
	float  B = LOAD_TENSOR(_Bias, uint2(0, pos.x/4))[pos.x%4]; // load here for less divergence
#if !defined(TRANSPOSE)
	if(kernel_size == 1) {
		// output == bias + torch.einsum("kj,ik->ij", input, weight)
		// modified from Linear.shader, WEIGHT_TRANSPOSED
		uint K = _WeightDim.y;
		for(uint k=threadId; k<K; k+=groupSize) {
			float4 A = LOAD_TENSOR(_Weight, uint2(pos.x, k));
			O += mul(A, float4x4(
				LOAD_TENSOR(_Input, uint2(k*4+0, pos.y)),
				LOAD_TENSOR(_Input, uint2(k*4+1, pos.y)),
				LOAD_TENSOR(_Input, uint2(k*4+2, pos.y)),
				LOAD_TENSOR(_Input, uint2(k*4+3, pos.y))));
		}
	} else {
		// output == bias + torch.einsum("kjp,ikq,pqr->ijr", input, weight, K) && K[p,q,r] == int(p==q+r*stride)
		// output[i,j][r] += input[k,j*P+p/4][p%4] * weight[i,k*Q+q/4][q%4], p = q+r*stride
		const uint P = (kernel_size+3*stride+3)/4;
		const uint Q = (kernel_size+3)/4;
		uint K = _InputDim.x;
		for(uint k=threadId; k<K; k+=groupSize) {
			float4 X[P], A[Q];
			{[unroll] for(uint p=0; p<P; p++)
				X[p] = LOAD_TENSOR(_Input, uint2(k, pos.y*P+p));}
			{[unroll] for(uint q=0; q<Q; q++)
				A[q] = LOAD_TENSOR(_Weight, uint2(pos.x, k*Q+q));}
			[unroll] for(uint q=0; q<kernel_size; q++) {
				uint4 p = q + stride*uint4(0,1,2,3);
				O += A[q/4][q%4] * float4(
					X[p[0]>>2][p[0]&3],
					X[p[1]>>2][p[1]&3],
					X[p[2]>>2][p[2]&3],
					X[p[3]>>2][p[3]&3]);
			}
		}
	}
#else
	// output == bias + torch.einsum("kjp,kiq,pqr->ijr", input, weight, K) && K[p,q,r] == int(q==pad+(1-p)*stride+r)
	// output[i,j*R+r/4][r%4] += input[k,j*P+p/4][p%4] * weight[k,i*Q+q/4][q%4], q = pad+(1-p)*stride+r
	const uint Q = (kernel_size+3)/4;
	const uint R = stride/2;
	const uint pad = (kernel_size-stride)/2;
#if defined(CONV_TRANSPOSE_K16_S8)
	const uint H = stride/4; // only 1/H weights are used when stride and pad are divisible by 4
#else
	const uint H = 1;
#endif
	uint K = _InputDim.x;
	uint j = pos.y / R;
	uint r = pos.y % R;
	uint rH = (pad/4 + r) % H;
	for(uint k=threadId; k<K; k+=groupSize) {
		float4 X = LOAD_TENSOR(_Input, uint2(k, j));
		float4 A[Q/H];
		{[unroll] for(uint q=0; q<Q/H; q++)
			A[q] = LOAD_TENSOR(_Weight, uint2(k, pos.x*Q+q*H+rH));}
		[unroll] for(uint p=0; p<4; p++) {
			int4 q = pad + (1-p)*stride + uint4(0,1,2,3);
			O += X[p] * (uint4(q) + r*4 >= kernel_size ? 0 : float4(
				A[min(Q-1, r+uint(q[0]>>2))/H][q[0]&3],
				A[min(Q-1, r+uint(q[1]>>2))/H][q[1]&3],
				A[min(Q-1, r+uint(q[2]>>2))/H][q[2]&3],
				A[min(Q-1, r+uint(q[3]>>2))/H][q[3]&3])); // handle negative q carefully
		}
	}
#endif
	O += B / groupSize; // accum here for less error
	return O;
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint4 pos = getThreadIdAndGroupSize(screenPos, _OutputDim);
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
#pragma shader_feature CONV_K1_S1 CONV_K3_S1 CONV_K5_S1 CONV_K7_S1 CONV_K11_S1\
	CONV_TRANSPOSE_K4_S2 CONV_TRANSPOSE_K16_S8
ENDHLSL
	}
}
}