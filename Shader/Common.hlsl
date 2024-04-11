#pragma warning (error : 3206) // implicit truncation of vector type

uint4 pcg4d(uint4 v) {
	// https://jcgt.org/published/0009/03/02/
	v = v * 1664525u + 1013904223u;
	v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	v ^= v >> 16u;
	v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	return v;
}
float4 erf(float4 x) {
	// cuda erff
	bool4 p = abs(x) >= +1.00295997e+00;
	float4 t = (p ? abs(x) : x * x);
	float4 r = (p ? +1.12198715e-04 : +8.48349446e-05);
	r = mad(r, t, (p ? -1.32752524e-03 : -8.21309164e-04));
	r = mad(r, t, (p ? +8.39653518e-03 : +5.21348882e-03));
	r = mad(r, t, (p ? -4.02465835e-02 : -2.68687736e-02));
	r = mad(r, t, (p ? +1.59504309e-01 : +1.12840049e-01));
	r = mad(r, t, (p ? +9.12917674e-01 : -3.76126647e-01));
	r = mad(r, t, (p ? +6.29060030e-01 : +1.28379151e-01));
	t = (p ? -abs(x) : x);
	r = mad(r, t, t);
	return (p ? sign(x) * (1 - exp2(r)) : r);
}
float4 gelu(float4 x) {
	// transformers.activations.GELUActivation
	return x * (0.5 + 0.5 * erf(x * 0.7071067811865475));
}
float4 gelu_tanh(float4 x) {
	// transformers.activations.NewGELUActivation
	float4 a = abs(x + x*x*x*0.044715) * 1.5957691216057308;
	float4 t = exp(-a) / (1.0 + exp(-a));
	return x * (x > 0 ? 1.0-t : t);
}
float4 silu(float4 x) {
	// torch.nn.functional.silu
	float4 a = abs(x);
	float4 t = exp(-a) / (1.0 + exp(-a));
	return x * (x > 0 ? 1.0-t : t);
}
float4 sigmoid(float4 x) {
	float4 a = abs(x);
	float4 t = exp(-a) / (1.0 + exp(-a));
	return (x > 0 ? 1.0-t : t);
}

float4 dequantizeWeight(float4 x, float offset) {
#if defined(WEIGHT_QUANTIZED_S24_Z8)
	// weight == (weight_u8 - zero_u8) / 256 * scale_f24
	return x*255 - offset;
#elif defined(WEIGHT_QUANTIZED_E8)
	// weight == (weight_u8 - (weight_u8 > max_u8 ? 255 : 0)) / 256 * exp2(exp_i8 * 0.5)
	return x*255 - (x > offset ? 255 : 0);
#else
	return x;
#endif
}
float4 dequantizeScale(float4 x, out float4 offset, float estep=2) {
#if defined(WEIGHT_QUANTIZED_S24_Z8)
	uint4 u32 = asuint(x);
	offset = u32 & 0xFF;
	return asfloat(u32 &~ 0xFF) / 256;
#elif defined(WEIGHT_QUANTIZED_E8)
	float4 i8 = round(x*255 - (x > 0.5 ? 256 : 0));
	float4 type = round(i8/85);
	offset = type * 0.25 + 0.5;
	return exp2((i8 - type*85) / estep - log2(256));
#else
	offset = 0;
	return 1;
#endif
}

void vertQuad(uint vertexID : SV_VertexID, out float4 posCS : SV_Position) {
	const float2 trigUV[3] = {{0,0},{0,1},{1,0}};
	// avoid inactive wave lanes by drawing fullscreen triangle 
	posCS = float4(trigUV[min(2,vertexID)]*4-1, 0, 1);
	#if UNITY_REVERSED_Z
		posCS.y *= -1;
	#endif
}

uint2 getThreadId(float4 screenPos, uint4 dim) {
	uint2 pos = floor(screenPos.yx);
	pos.y = (pos.y<<dim.z) | (pos.x & ((1<<dim.z)-1));
	pos.x >>= dim.z;
	return pos;
}
uint4 getThreadIdAndGroupSize(float4 screenPos, uint4 dim) {
	uint2 pos = floor(screenPos.yx);
	uint threadId = dot(pos & ((1<<dim.w)-1), uint2(1, 1<<dim.w));
	uint groupSize = (1<<dim.w)<<dim.w;
	pos >>= dim.w;
	pos.y = (pos.y<<dim.z) | (pos.x & ((1<<dim.z)-1));
	pos.x >>= dim.z;
	return uint4(pos, threadId, groupSize);
}

float4 loadTensor(Texture2D<float4> tex, uint4 dim, uint2 ij) {
	return tex.mips[dim.w][uint2(ij.y>>dim.z, (ij.y & ((1<<dim.z)-1)) | (ij.x<<dim.z))];
}
#define LOAD_TENSOR(name, ij) loadTensor(name##Tex, name##Dim, ij)