Shader "GPT/Scatter" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (1, 1, 0, 0)
	_WeightDim("_WeightDim", Vector) = (0, 0, 0, 0) // zero = use constant weight
	[HideInInspector]_OutputTex("_OutputTex", 2D) = "black" {}
	[NoScaleOffset]  _InputTex ("_InputTex",  2D) = "black" {}
	[NoScaleOffset]  _WeightTex("_WeightTex", 2D) = "black" {}
	_InputOff ("_InputOff",  Vector) = (0, 0, 0, 0)
	_InputChan("_InputChan", Int) = 0
	_ColorMask("_ColorMask", Int) = 15
	_Weight   ("_Weight",    Vector) = (0, 0, 0, 0)
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
Texture2D<float4> _InputTex;  uint4 _InputDim;
Texture2D<float4> _WeightTex; uint4 _WeightDim;
uniform uint2 _InputOff;
uniform uint _InputChan;
uniform uint _ColorMask;
uniform float4 _Weight;

struct GeomInput {};
struct GeomOutput { float4 posCS : SV_Position; };
void vert() {}

static const uint batchSize = 4096;
static const uint instanceCount = 32;
static const uint geometryCount = 64;
[maxvertexcount(geometryCount*4)]
[instance(instanceCount)]
void geom(triangle GeomInput input[3], inout TriangleStream<GeomOutput> stream, uint instanceId : SV_GSInstanceID, uint primitiveId : SV_PrimitiveID) {
	GeomOutput o;
	for(uint i = 0; i < geometryCount; i++) {
		uint src = (primitiveId * instanceCount + instanceId) * geometryCount + i;
		if(src >= batchSize)
			return;
		uint payload = src;
		float4 aabb = float4(0,0,1,1);
#ifdef WEIGHT_TRANSPOSED
		if(any(_InputOff+uint2(_InputChan, src/4) >= _InputDim.xy))
			return;
		uint dst = LOAD_TENSOR(_Input, _InputOff+uint2(_InputChan, src/4))[src%4];
		if(_ColorMask != (1<<(3-dst%4))) // RGBA
			continue;
		aabb.xz = (float2(0,1) + (dst/4 >> _OutputDim.z)) / (_OutputDim.y>>_OutputDim.z);
		payload = (payload<<_OutputDim.z) | (dst/4 & ((1<<_OutputDim.z)-1));
#else
		if(any(_InputOff+uint2(src, _InputChan/4) >= _InputDim.xy))
			return;
		uint dst = LOAD_TENSOR(_Input, _InputOff+uint2(src, _InputChan/4))[_InputChan%4];
		aabb.yw = (float2(0,1) + dst) / _OutputDim.x;
#endif
		aabb = aabb*2-1;
		#if UNITY_REVERSED_Z
			aabb.yw *= -1;
		#endif
		o.posCS = float4(aabb.xy, payload/65536.0, 1); stream.Append(o);
		o.posCS = float4(aabb.xw, payload/65536.0, 1); stream.Append(o);
		o.posCS = float4(aabb.zy, payload/65536.0, 1); stream.Append(o);
		o.posCS = float4(aabb.zw, payload/65536.0, 1); stream.Append(o);
		stream.RestartStrip();
	}
}
float4 main(uint2 pos, uint payload) {
	// torch.Tensor.scatter_()
	// (transpose ? output[i][input[c][j]] : output[input[i][c]][j]) = weight[i][j]

#ifdef WEIGHT_TRANSPOSED
	if((pos.y-payload) & ((1<<_OutputDim.z)-1))
		discard;
	uint src = payload>>_OutputDim.z;
	return _WeightDim.x ? LOAD_TENSOR(_Weight, _InputOff+uint2(pos.x, src/4))[src%4] : _Weight;
#else
	uint src = payload;
	return _WeightDim.x ? LOAD_TENSOR(_Weight, _InputOff+uint2(src, pos.y)) : _Weight;
#endif
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint2 pos = getThreadId(screenPos, _OutputDim);
	// TODO: payload might be wrong on non-dx platform
	return main(pos, round(screenPos.z*65536.0));
}
ENDHLSL
	Pass {
		Cull Off
		ColorMask [_ColorMask]
HLSLPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma geometry geom
#pragma fragment frag
#pragma shader_feature WEIGHT_TRANSPOSED
ENDHLSL
	}
}
}