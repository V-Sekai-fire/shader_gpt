Shader "GPT/Scatter" {
Properties {
	_OutputDim("_OutputDim", Vector) = (1, 1, 0, 0)
	_InputDim ("_InputDim",  Vector) = (0, 0, 0, 0) // zero = use constant
	_IndexDim ("_IndexDim",  Vector) = (1, 1, 0, 0)
	[HideInInspector]
	_OutputTex("_OutputTex", 2D) = "black" {}
	_InputTex ("_InputTex",  2D) = "black" {}
	_IndexTex ("_IndexTex",  2D) = "black" {}
	_Input    ("_Input",     Vector) = (0, 0, 0, 0)
	_IndexOff ("_IndexOff",  Vector) = (0, 0, 0, 0)
	_IndexChan("_IndexChan", Int) = 0
	_ColorMask("_ColorMask", Int) = 15
}
SubShader {
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"

uint4 _OutputDim;
DEFINE_TEXTURE2D(_InputTex); uint4 _InputDim;
DEFINE_TEXTURE2D(_IndexTex); uint4 _IndexDim;
uniform float4 _Input;
uniform uint _IndexChan;
uniform uint _BatchOff;
uniform uint _ColorMask;

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
		src += _BatchOff;
		uint payload = src;
		float4 aabb = float4(0,0,1,1);
#ifdef AXIS_LAST
		if(src/4 >= _IndexDim.y)
			return;
		uint dst = LOAD_TENSOR(_Index, uint2(_IndexChan, src/4))[src%4];
		if(_ColorMask != (1<<(3-dst%4))) // RGBA
			continue;
		aabb.xz = (float2(0,1) + (dst/4 >> _OutputDim.z)) / (_OutputDim.y>>_OutputDim.z);
		payload = (payload<<_OutputDim.z) | (dst/4 & ((1<<_OutputDim.z)-1));
#else
		if(src >= _IndexDim.x)
			return;
		uint dst = LOAD_TENSOR(_Index, uint2(src, _IndexChan/4))[_IndexChan%4];
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
	// torch.Tensor.index_copy_(axis, index, input)
	// (axis == 1 ? output[i,index[c,j][jj]/4][index[c,j][jj]%4] : output[index[i,c/4][c%4],j][jj]) = input[i,j][jj]

#ifdef AXIS_LAST
	if((pos.y-payload) & ((1<<_OutputDim.z)-1))
		discard;
	uint src = payload>>_OutputDim.z;
	return _InputDim.x ? LOAD_TENSOR(_Input, uint2(pos.x, src/4))[src%4] : _Input;
#else
	uint src = payload;
	return _InputDim.x ? LOAD_TENSOR(_Input, uint2(src, pos.y)) : _Input;
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
#pragma shader_feature AXIS_LAST
ENDHLSL
	}
}
}