uint4 pcg4d(uint4 v) {
	// https://jcgt.org/published/0009/03/02/
	v = v * 1664525u + 1013904223u;
	v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	v ^= v >> 16u;
	v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	return v;
}
float4 erf(float4 v) {
	// https://github.com/microsoft/onnxruntime/blob/main/js/web/lib/wasm/jsep/webgpu/ops/unary-op.ts
	const float r0 = 0.3275911;
	const float r1 = 0.254829592;
	const float r2 = -0.284496736;
	const float r3 = 1.421413741;
	const float r4 = -1.453152027;
	const float r5 = 1.061405429;
	float4 x = 1.0 / (1.0 + r0 * abs(v));
	return sign(v) * (1.0 - ((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x * exp(-v*v));
}
float4 gelu(float4 x) {
	// transformers.activations.GELUActivation
	return x * (0.5 + 0.5 * erf(x * 0.7071067811865475));
}
float4 gelu_new(float4 x) {
	// transformers.activations.NewGELUActivation
	float4 a = (x + x*x*x*0.044715) * 1.5957691216057308;
	float4 t = 1.0 / (1.0 + exp(-abs(a)));
	return x * (x > 0 ? t : 1.0-t);
}

float4 dequantizeWeight(float4 x, float offset) {
#ifdef QUANTIZE_WEIGHT
	return x - (x > offset ? 1 : 0);
#else
	return x;
#endif
}
float4 dequantizeScale(float4 x, out float4 offset, bool enabled=true, float dstep=256, float estep=2) {
#ifdef QUANTIZE_WEIGHT
	if(enabled) {
		float4 byte = round(x*255 - (x > 0.5 ? 256 : 0));
		float4 type = round(byte/85);
		offset = type * 0.25 + 0.5;
		return exp2((byte - type*85) / estep + log2(255/dstep));
	}
#endif
	offset = asfloat(0x7f7fffff); // infinity-1
	return 1;
}

void vertQuad(float2 uv : TEXCOORD0, out float4 vertex : SV_Position) {
	vertex = float4(uv*2-1, UNITY_NEAR_CLIP_VALUE, 1);
}
uint2 getGroupThreadIdAndSize(inout uint2 pos, uint lod) {
	uint threadId = dot(pos & ((1<<lod)-1), uint2(1, 1<<lod));
	pos >>= lod;
	return uint2(threadId, (1<<lod)<<lod);
}