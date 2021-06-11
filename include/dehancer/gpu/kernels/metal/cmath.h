#pragma once

#include <metal_stdlib>

using namespace metal;

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline static  float2 __attribute__((overloadable)) make_float2(float x, float y) {
  return float2(x, y);
}

inline static  float3 __attribute__((overloadable)) make_float3(float x, float y, float z) {
  return float3(x, y, z);
}

inline static  float4 __attribute__((overloadable)) make_float4(float x, float y, float z, float w) {
  return float4(x, y, z, w);
}

inline static  int2 __attribute__((overloadable)) make_int2(int x, int y) {
  return int2(x, y);
}

inline static  int3 __attribute__((overloadable)) make_int3(int x, int y, int z) {
  return int3(x, y, z);
}

inline static  int4 __attribute__((overloadable)) make_int4(int x, int y, int z, int w) {
  return int4(x, y, z, w);
}

inline static  uint2 __attribute__((overloadable)) make_uint2(uint x, uint y) {
  return uint2(x, y);
}

inline static  uint3 __attribute__((overloadable)) make_uint3(uint x, uint y, uint z) {
  return uint3(x, y, z);
}

inline static  uint4 __attribute__((overloadable)) make_uint4(uint x, uint y, uint z, uint w) {
  return uint4(x, y, z, w);
}


inline static  float2 __attribute__((overloadable)) make_float2(float s) {
  return float2(s, s);
}

inline static  float2 __attribute__((overloadable)) make_float2(float3 a) {
  return float2(a.x, a.y);
}

inline static  float2 __attribute__((overloadable)) make_float2(float4 a) {
  return float2(a.x, a.y);
}

inline static  float2 __attribute__((overloadable)) make_float2(int2 a) {
  return float2(float(a.x), float(a.y));
}

inline static  float2 __attribute__((overloadable)) make_float2(uint2 a) {
  return float2(float(a.x), float(a.y));
}

inline static  int2 __attribute__((overloadable)) make_int2(int s) {
  return int2(s, s);
}

inline static  int2 __attribute__((overloadable)) make_int2(int3 a) {
  return int2(a.x, a.y);
}

inline static  int2 __attribute__((overloadable)) make_int2(uint2 a) {
  return int2(int(a.x), int(a.y));
}

inline static  int2 __attribute__((overloadable)) make_int2(float2 a) {
  return int2(int(a.x), int(a.y));
}

inline static  uint2 __attribute__((overloadable)) make_uint2(uint s) {
  return uint2(s, s);
}

inline static  uint2 __attribute__((overloadable)) make_uint2(uint3 a) {
  return uint2(a.x, a.y);
}

inline static  uint2 __attribute__((overloadable)) make_uint2(int2 a) {
  return uint2(uint(a.x), uint(a.y));
}

inline static  float3 __attribute__((overloadable)) make_float3(float s) {
  return float3(s, s, s);
}

inline static  float3 __attribute__((overloadable)) make_float3(float2 a) {
  return float3(a.x, a.y, 0.0f);
}

inline static  float3 __attribute__((overloadable)) make_float3(float2 a, float s) {
  return float3(a.x, a.y, s);
}

inline static  float3 __attribute__((overloadable)) make_float3(float4 a) {
  return float3(a.x, a.y, a.z);
}

inline static  float3 __attribute__((overloadable)) make_float3(int3 a) {
  return float3(float(a.x), float(a.y), float(a.z));
}

inline static  float3 __attribute__((overloadable)) make_float3(uint3 a) {
  return float3((float)(a.x), (float)(a.y), (float)(a.z));
}

inline static  int3 __attribute__((overloadable)) make_int3(int s) {
  return int3(s, s, s);
}

inline static  int3 __attribute__((overloadable)) make_int3(int2 a) {
  return int3(a.x, a.y, 0);
}

inline static  int3 __attribute__((overloadable)) make_int3(int2 a, int s) {
  return int3(a.x, a.y, s);
}

inline static  int3 __attribute__((overloadable)) make_int3(uint3 a) {
  return int3((int)(a.x), (int)(a.y), (int)(a.z));
}

inline static  int3 __attribute__((overloadable)) make_int3(float3 a) {
  return int3((int)(a.x), (int)(a.y), (int)(a.z));
}

inline static  uint3 __attribute__((overloadable)) make_uint3(uint s) {
  return uint3(s, s, s);
}

inline static  uint3 __attribute__((overloadable)) make_uint3(uint2 a) {
  return uint3(a.x, a.y, 0);
}

inline static  uint3 __attribute__((overloadable)) make_uint3(uint2 a, uint s) {
  return uint3(a.x, a.y, s);
}

inline static  uint3 __attribute__((overloadable)) make_uint3(uint4 a) {
  return uint3(a.x, a.y, a.z);
}

inline static  uint3 __attribute__((overloadable)) make_uint3(int3 a) {
  return uint3((uint)(a.x), (uint)(a.y), (uint)(a.z));
}

inline static  float4 __attribute__((overloadable)) make_float4(float s) {
  return float4(s, s, s, s);
}

inline static  float4 __attribute__((overloadable)) make_float4(float3 a) {
  return float4(a.x, a.y, a.z, 0.0f);
}

inline static  float4 __attribute__((overloadable)) make_float4(float3 a, float w) {
  return float4(a.x, a.y, a.z, w);
}

inline static  float4 __attribute__((overloadable)) make_float4(int4 a) {
  return float4((float)(a.x), (float)(a.y), (float)(a.z), (float)(a.w));
}

inline static  float4 __attribute__((overloadable)) make_float4(uint4 a) {
  return float4((float)(a.x), (float)(a.y), (float)(a.z), (float)(a.w));
}

inline static  int4 __attribute__((overloadable)) make_int4(int s) {
  return int4(s, s, s, s);
}

inline static  int4 __attribute__((overloadable)) make_int4(int3 a) {
  return int4(a.x, a.y, a.z, 0);
}

inline static  int4 __attribute__((overloadable)) make_int4(int3 a, int w) {
  return int4(a.x, a.y, a.z, w);
}

inline static  int4 __attribute__((overloadable)) make_int4(uint4 a) {
  return int4((int)(a.x), (int)(a.y), (int)(a.z), (int)(a.w));
}

inline static  int4 __attribute__((overloadable)) make_int4(float4 a) {
  return int4((int)(a.x), (int)(a.y), (int)(a.z), (int)(a.w));
}


inline static  uint4 __attribute__((overloadable)) make_uint4(uint s) {
  return uint4(s, s, s, s);
}

inline static  uint4 __attribute__((overloadable)) make_uint4(uint3 a) {
  return uint4(a.x, a.y, a.z, 0);
}

inline static  uint4 __attribute__((overloadable)) make_uint4(uint3 a, uint w) {
  return uint4(a.x, a.y, a.z, w);
}

inline static  uint4 __attribute__((overloadable)) make_uint4(int4 a) {
  return uint4((uint)(a.x), (uint)(a.y), (uint)(a.z), (uint)(a.w));
}

//
//////////////////////////////////////////////////////////////////////////////////
//// min
//////////////////////////////////////////////////////////////////////////////////

inline static float __attribute__((overloadable)) fminf(float a, float b) {
  return metal::min(a, b);
}

inline static   float2 __attribute__((overloadable)) fminf(float2 a, float2 b) {
  return metal::min(a, b); //make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

inline static  float3 __attribute__((overloadable)) fminf(float3 a, float3 b) {
  return metal::min(a, b); //make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

inline static   float4 __attribute__((overloadable)) fminf(float4 a, float4 b) {
  return metal::min(a, b); //make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

//////////////////////////////////////////////////////////////////////////////////
//// max
//////////////////////////////////////////////////////////////////////////////////

inline static  float __attribute__((overloadable)) fmaxf(float a, float b) {
  return metal::max(a, b);
}

inline static  float2 __attribute__((overloadable)) fmaxf(float2 a, float2 b) {
  return metal::max(a, b); //make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

inline static  float3 __attribute__((overloadable)) fmaxf(float3 a, float3 b) {
  return metal::max(a, b); //make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline static  float4 __attribute__((overloadable)) fmaxf(float4 a, float4 b) {
  return metal::max(a, b); //make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline static  float __attribute__((overloadable)) lerp(float a, float b, float t) {
  return a + t * (b - a);
}

inline static  float2 __attribute__((overloadable)) lerp(float2 a, float2 b, float t) {
  return a + t * (b - a);
}

inline static  float3 __attribute__((overloadable)) lerp(float3 a, float3 b, float t) {
  return a + t * (b - a);
}

inline static  float4 __attribute__((overloadable)) lerp(float4 a, float4 b, float t) {
  return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline static  float __attribute__((overloadable)) floorf(float v) {
  return floor(v);
}

inline static float2 __attribute__((overloadable)) floorf(float2 v) {
  return floor(v);
}

inline static  float3 __attribute__((overloadable)) floorf(float3 v) {
  return floor(v);
}

////////////////////////////////////////////////////////////////////////////////
// ceil
////////////////////////////////////////////////////////////////////////////////

inline static  float __attribute__((overloadable)) ceilf(float v) {
  return ceil(v);
}

inline static float2 __attribute__((overloadable)) ceilf(float2 v) {
  return ceil(v);
}

inline static  float3 __attribute__((overloadable)) ceilf(float3 v) {
  return ceil(v);
}

inline static  float4 __attribute__((overloadable)) ceilf(float4 v) {
  return ceil(v);
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline static  float __attribute__((overloadable)) fracf(float v) {
  return metal::fract(v);
}

inline static  float2 __attribute__((overloadable)) fracf(float2 v) {
  return metal::fract(v);
}

inline static  float3 __attribute__((overloadable)) fracf(float3 v) {
  return metal::fract(v);
}

inline static  float4 __attribute__((overloadable)) fracf(float4 v) {
  return metal::fract(v);
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline static  float __attribute__((overloadable)) fmodf(float a, float b) {
  return metal::fmod(a,b);
}

inline static  float2 __attribute__((overloadable)) fmodf(float2 a, float2 b) {
  return metal::fmod(a,b);//make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}

inline static  float3 __attribute__((overloadable)) fmodf(float3 a, float3 b) {
  return metal::fmod(a,b);//make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}

inline static  float4 __attribute__((overloadable)) fmodf(float4 a, float4 b) {
  return metal::fmod(a,b);//make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}


////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline static  float3 __attribute__((overloadable)) reflect(float3 i, float3 n) {
  return i - 2.0f * n * dot(n, i);
}

#define powf pow
#define roundf round
#define log2f log2
#define log10f log10

