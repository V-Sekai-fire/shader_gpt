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
float4 gelu_new(float4 x) {
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
float4 dequantizeScale(float4 x, out float4 offset, bool enabled=true, float estep=2) {
#if defined(WEIGHT_QUANTIZED_S24_Z8)
	if(enabled) {
		uint4 u32 = asuint(x);
		offset = u32 & 0xFF;
		return asfloat(u32 &~ 0xFF) / 256;
	}
	offset = 0;
	return 1.0/255;
#elif defined(WEIGHT_QUANTIZED_E8)
	if(enabled) {
		float4 i8 = round(x*255 - (x > 0.5 ? 256 : 0));
		float4 type = round(i8/85);
		offset = type * 0.25 + 0.5;
		return exp2((i8 - type*85) / estep - log2(256));
	}
	offset = asfloat(0x7f7fffff); // infinity-1
	return 1.0/255;
#else
	offset = 0;
	return 1;
#endif
}

void vertQuad(float2 uv : TEXCOORD0, out float4 vertex : SV_Position) {
	vertex = float4(uv*2-1, UNITY_NEAR_CLIP_VALUE, 1);
}
uint2 getThreadId(float4 screenPos) {
	return floor(screenPos.yx);
}
uint2 getThreadId(float4 screenPos, uint4 dim) {
	uint2 pos = getThreadId(screenPos);
	pos.y += pos.x % dim.z * 16384;
	pos.x /= dim.z;
	return pos;
}
uint4 getThreadIdAndGroupSize(float4 screenPos, uint4 dim) {
	uint2 pos = getThreadId(screenPos);
	uint threadId = dot(pos & ((1<<dim.w)-1), uint2(1, 1<<dim.w));
	uint groupSize = (1<<dim.w)<<dim.w;
	pos >>= dim.w;
	pos.y += pos.x % dim.z * 16384;
	pos.x /= dim.z;
	return uint4(pos, threadId, groupSize);
}

float4 loadTensor(Texture2D<float4> tex, uint i, uint j, uint lod=0) {
	return tex.mips[lod][uint2(j,i)];
}
float4 loadTensor(Texture2D<float4> tex, uint i, uint j, uint4 dim) {
	return tex.mips[dim.w][uint2(j%16384,i*dim.z+j/16384)];
}
float4 loadTensor(Texture2D<float4> tex, uint2 ij) {
	return loadTensor(tex, ij.x, ij.y);
}
float4 loadTensor(Texture2D<float4> tex, uint2 ij, uint4 dim) {
	return loadTensor(tex, ij.x, ij.y, dim);
}