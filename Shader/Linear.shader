Shader "GPT/Linear" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 1, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 1, 0)
	[HideInInspector]_OutputTex("_OutputTex",2D) = "black" {}
	[NoScaleOffset] _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset] _WeightTex("_WeightTex", 2D) = "black" {}
	[NoScaleOffset] _ScaleTex ("_ScaleTex",  2D) = "black" {}
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
Texture2D<float4> _ScaleTex;
uniform uint _Head;

float4 main(uint2 pos, uint threadId, uint groupSize) {
	// torch.nn.functional.linear(bias=None) with multi-head support
	// weight[i,j] *= scale[i/4,j][i%4]
	// output[i,h*J+j][jj] += input[i,h*K+k][kk] * (transpose ? weight[k*4+kk,h*J+j][jj] : weight[j*4+jj,h*K+k][kk])

	uint J = _OutputDim.y/_Head, j = pos.y%J;
	uint K = _InputDim.y/_Head, hK = pos.y/J*K;
	float4 O = 0;
	for(uint k=threadId; k<K; k+=groupSize) {
		float4 X = _InputTex.mips[_InputDim.w][uint2(pos.x,hK+k).yx];
		#ifdef WEIGHT_TRANSPOSED
			float4 offset, scale = dequantizeScale(_ScaleTex[uint2(k,pos.y).yx], offset);
			O += mul(scale * X, float4x4(
				dequantizeWeight(_WeightTex[uint2(k*4+0,pos.y).yx], offset[0]),
				dequantizeWeight(_WeightTex[uint2(k*4+1,pos.y).yx], offset[1]),
				dequantizeWeight(_WeightTex[uint2(k*4+2,pos.y).yx], offset[2]),
				dequantizeWeight(_WeightTex[uint2(k*4+3,pos.y).yx], offset[3])));
		#else
			// tested: error rate of per-out-channel block q8 is 10%~50% smaller than per-input-channel
			// awq does this too: github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization/gemv_cuda.cu
			float4 offset, scale = dequantizeScale(_ScaleTex[uint2(j,hK+k).yx], offset);
			O += scale * mul(float4x4(
				dequantizeWeight(_WeightTex[uint2(j*4+0,hK+k).yx], offset[0]),
				dequantizeWeight(_WeightTex[uint2(j*4+1,hK+k).yx], offset[1]),
				dequantizeWeight(_WeightTex[uint2(j*4+2,hK+k).yx], offset[2]),
				dequantizeWeight(_WeightTex[uint2(j*4+3,hK+k).yx], offset[3])), X);
		#endif
	}
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
#pragma shader_feature WEIGHT_TRANSPOSED
#pragma shader_feature WEIGHT_QUANTIZED
ENDHLSL
	}
}
}