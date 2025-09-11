#pragma once

#include <metal_stdlib>

using namespace metal;
#include <dehancer/small_vectors.h>

//////////////////////////////////////////////////////////////////////////////////
//// min
//////////////////////////////////////////////////////////////////////////////////

inline static float __attribute__((overloadable)) fminf(float a, float b) {
  return metal::min(a, b);
}

inline static   float2 __attribute__((overloadable)) fminf(float2 a, float2 b) {
  return metal::min(a, b);
}

inline static  float3 __attribute__((overloadable)) fminf(float3 a, float3 b) {
  return metal::min(a, b);
}

inline static   float4 __attribute__((overloadable)) fminf(float4 a, float4 b) {
  return metal::min(a, b);
}

//////////////////////////////////////////////////////////////////////////////////
//// max
//////////////////////////////////////////////////////////////////////////////////

inline static  float __attribute__((overloadable)) fmaxf(float a, float b) {
  return metal::max(a, b);
}

inline static  float2 __attribute__((overloadable)) fmaxf(float2 a, float2 b) {
  return metal::max(a, b);
}

inline static  float3 __attribute__((overloadable)) fmaxf(float3 a, float3 b) {
  return metal::max(a, b);
}

inline static  float4 __attribute__((overloadable)) fmaxf(float4 a, float4 b) {
  return metal::max(a, b);
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

