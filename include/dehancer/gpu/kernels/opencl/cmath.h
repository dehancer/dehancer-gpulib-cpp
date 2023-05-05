/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    This file implements common mathematical operations on vector types
    (float3, float4 etc.) since these are not provided as standard by CUDA.

    The syntax is modelled on the Cg standard library.

    This is part of the CUTIL library and is not supported by NVIDIA.

    Thanks to Linh Hah for additions and fixes.
*/


#ifndef DEHANCER_GPULIB_CMATH_OPENCL_H


////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////


/* make_type definitions with opencl style element initializers */
#ifdef make_float2
#  undef make_float2
#endif
#ifdef make_float3
#  undef make_float3
#endif
#ifdef make_float4
#  undef make_float4
#endif
#ifdef make_int2
#  undef make_int2
#endif
#ifdef make_int3
#  undef make_int3
#endif
#ifdef make_int4
#  undef make_int4
#endif
#ifdef make_uint2
#  undef make_uint2
#endif
#ifdef make_uint3
#  undef make_uint3
#endif
#ifdef make_uint4
#  undef make_uint4
#endif
#ifdef make_uchar4
#  undef make_uchar4
#endif

#define make_float2(x, y)        ((float2){x, y})
#define make_float3(x, y, z)     ((float3){x, y, z})
#define make_float4(x, y, z, w)  ((float4){x, y, z, w})
#define make_int2(x, y)          ((int2){x, y})
#define make_int3(x, y, z)       ((int3){x, y, z})
#define make_int4(x, y, z, w)    ((int4){x, y, z, w})
#define make_uint2(x, y)         ((uint2){x, y})
#define make_uint3(x, y, z)      ((uint3){x, y, z})
#define make_uint4(x, y, z, w)   ((uint4){x, y, z, w})
#define make_uchar4(x, y, z, w)  ((uchar4){x, y, z, w})

/***
 * TODO: float2x2,float3x3,float4x4 constructors
 *
 * @param r0
 * @param r1
 * @param r2
 * @return
 */
static inline float3x3 __attribute__((overloadable)) make_float3x3(float3 r0, float3 r1, float3 r2) {
 
  float3x3 val;
  
#if defined(ARMA_INCLUDES)
  val(0,0)=r0.x(); val(0,1)=r0.y(); val(0,2)=r0.z();
  val(1,0)=r1.x(); val(1,1)=r1.y(); val(1,2)=r1.z();
  val(2,0)=r2.x(); val(2,1)=r2.y(); val(2,2)=r2.z();
#else
  val.v[0].x=r0.x; val.v[0].y=r0.y; val.v[0].z=r0.z;
  val.v[1].x=r1.x; val.v[1].y=r1.y; val.v[1].z=r1.z;
  val.v[2].x=r2.x; val.v[2].y=r2.y; val.v[2].z=r2.z;
#endif
  return val;
}

static inline float3 __attribute__((overloadable)) matrix3x3_mul(float3x3 m, float3 v) {
  #if defined(ARMA_INCLUDES)
  return m * v;
  #else
  return make_float3(
          m.v[0].x * v.x + m.v[0].y * v.y + m.v[0].z * v.z,
          m.v[1].x * v.x + m.v[1].y * v.y + m.v[1].z * v.z,
          m.v[2].x * v.x + m.v[2].y * v.y + m.v[2].z * v.z
  );
  #endif
}

static inline float4 __attribute__((overloadable)) matrix4x4_mul(float4x4 m, float4 v) {
//  #if defined(ARMA_INCLUDES)
  return v;
//  #else
//  return make_float4(
//          m.v[0].x * v.x + m.v[0].y * v.y + m.v[0].z * v.z + m.v[0].w * v.w,
//          m.v[1].x * v.x + m.v[1].y * v.y + m.v[1].z * v.z + m.v[1].w * v.w,
//          m.v[2].x * v.x + m.v[2].y * v.y + m.v[2].z * v.z + m.v[2].w * v.w,
//          m.v[3].x * v.x + m.v[3].y * v.y + m.v[3].z * v.z + m.v[3].w * v.w
//  );
//  #endif
}

//
//////////////////////////////////////////////////////////////////////////////////
//// min
//////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fminf(float a, float b) {
  return fmin(a, b);
}

static inline  float2 __attribute__((overloadable)) fminf(float2 a, float2 b) {
#if defined(ARMA_INCLUDES)
  return float2({fminf(a.x(), b.x()), fminf(a.y(), b.y())});
#else
  return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
#endif
}

static inline float3 __attribute__((overloadable)) fminf(float3 a, float3 b) {
#if defined(ARMA_INCLUDES)
  return float3({fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z())});
#else
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
#endif
}

static inline  float4 __attribute__((overloadable)) fminf(float4 a, float4 b) {
#if defined(ARMA_INCLUDES)
  return float4({fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z()), fminf(a.w(), b.w())});
#else
  return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
#endif
}

//////////////////////////////////////////////////////////////////////////////////
//// max
//////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fmaxf(float a, float b) {
  return fmax(a, b);
}

static inline float2 __attribute__((overloadable)) fmaxf(float2 a, float2 b) {
#if defined(ARMA_INCLUDES)
  return float2({fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y())});
#else
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
#endif
}

static inline float3 __attribute__((overloadable)) fmaxf(float3 a, float3 b) {
#if defined(ARMA_INCLUDES)
  return float3({fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z())});
#else
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
#endif
}

static inline float4 __attribute__((overloadable)) fmaxf(float4 a, float4 b) {
#if defined(ARMA_INCLUDES)
  return float4({fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z()), fmaxf(a.w(), b.w())});
#else
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) lerp(float a, float b, float t) {
  return a + t * (b - a);
}

static inline float2 __attribute__((overloadable)) lerp(float2 a, float2 b, float t) {
  return a + t * (b - a);
}

static inline float3 __attribute__((overloadable)) lerp(float3 a, float3 b, float t) {
  return a + t * (b - a);
}

static inline float4 __attribute__((overloadable)) lerp(float4 a, float4 b, float t) {
  return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) floorf(float v) {
  return floor(v);
}

static inline float2 __attribute__((overloadable)) floorf(float2 v) {
#if defined(ARMA_INCLUDES)
  return float2({floorf(v.x()), floorf(v.y())});
#else
  return make_float2(floorf(v.x), floorf(v.y));
#endif

}

static inline float3 __attribute__((overloadable)) floorf(float3 v) {
#if defined(ARMA_INCLUDES)
  return float2({floorf(v.x()), floorf(v.y()), floorf(v.z())});
#else
  return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
#endif
}

static inline float4 __attribute__((overloadable)) floorf(float4 v) {
#if defined(ARMA_INCLUDES)
  return float4({floorf(v.x()), floorf(v.y()), floorf(v.z()), floorf(v.w())});
#else
  return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
#endif
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fracf(float v) {
  return v - floorf(v);
}

static inline float2 __attribute__((overloadable)) fracf(float2 v) {
#if defined(ARMA_INCLUDES)
  return float2({fracf(v.x()), fracf(v.y())});
#else
  return make_float2(fracf(v.x), fracf(v.y));
#endif
}

static inline float3 __attribute__((overloadable)) fracf(float3 v) {
  #if defined(ARMA_INCLUDES)
  return float2({fracf(v.x()), fracf(v.y()), fracf(v.z())});
  #else
  return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
  #endif
}

static inline float4 __attribute__((overloadable)) fracf(float4 v) {
  #if defined(ARMA_INCLUDES)
  return float2({fracf(v.x()), fracf(v.y()), fracf(v.z()), fracf(v.w())});
  #else
  return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
  #endif
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fmodf(float a, float b) {
  return fmod(a,b);
}

static inline float2 __attribute__((overloadable)) fmodf(float2 a, float2 b) {
  #if defined(ARMA_INCLUDES)
  return float2({fmodf(a.x(), b.x()), fmodf(a.y(), b.y())});
  #else
  return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
  #endif
}

static inline float3 __attribute__((overloadable)) fmodf(float3 a, float3 b) {
  #if defined(ARMA_INCLUDES)
  return float2({fmodf(a.x(), b.x()), fmodf(a.y(), b.y()), fmodf(a.z(), b.z())});
  #else
  return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
  #endif
}

static inline float4 __attribute__((overloadable)) fmodf(float4 a, float4 b) {
  #if defined(ARMA_INCLUDES)
  return float2({fmodf(a.x(), b.x()), fmodf(a.y(), b.y()), fmodf(a.z(), b.z()), fmodf(a.w(), b.w())});
  #else
  return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
  #endif
}


////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

static inline float3 __attribute__((overloadable)) reflect(float3 i, float3 n) {
  return i - 2.0f * n * dot(n, i);
}

#define roundf round
#define ceilf ceil

#endif
