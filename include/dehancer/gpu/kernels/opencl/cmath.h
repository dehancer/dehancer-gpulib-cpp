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

//static inline float2 __attribute__((overloadable)) make_float2(float x, float y) {
//  return (float2){x, y};
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(float x, float y, float z) {
//  return (float3){x, y, z};
//}
//
//static inline float4 __attribute__((overloadable)) make_float4(float x, float y, float z, float w) {
//  return (float4){x, y, z, w};
//}
//
//static inline int2 __attribute__((overloadable)) make_int2(int x, int y) {
//  return (int2){x, y};
//}
//
//static inline int3 __attribute__((overloadable)) make_int3(int x, int y, int z) {
//  return (int3){x, y, z};
//}
//
//static inline int4 __attribute__((overloadable)) make_int4(int x, int y, int z, int w) {
//  return (int4){x, y, z, w};
//}
//
//static inline uint2 __attribute__((overloadable)) make_uint2(uint x, uint y) {
//  return (uint2){x, y};
//}
//
//static inline uint3 __attribute__((overloadable)) make_uint3(uint x, uint y, uint z) {
//  return (uint3){x, y, z};
//}
//
//static inline uint4 __attribute__((overloadable)) make_uint4(uint x, uint y, uint z, uint w) {
//  return (uint4){x, y, z, w};
//}
//
//
//static inline float2 __attribute__((overloadable)) make_float2(float s) {
//  return make_float2(s, s);
//}
//
//static inline float2 __attribute__((overloadable)) make_float2(float3 a) {
//  return make_float2(a.x, a.y};
//}
//
//static inline float2 __attribute__((overloadable)) make_float2(float4 a) {
//  return make_float2(a.x, a.y);
//}
//
//static inline float2 __attribute__((overloadable)) make_float2(int2 a) {
//  return make_float2((float)(a.x), (float)(a.y));
//}
//
//static inline float2 __attribute__((overloadable)) make_float2(uint2 a) {
//  return make_float2((float)(a.x), (float)(a.y));
//}
//
//static inline int2 __attribute__((overloadable)) make_int2(int s) {
//  return make_int2(s, s);
//}
//
//static inline int2 __attribute__((overloadable)) make_int2(int3 a) {
//  return make_int2(a.x, a.y);
//}
//
//static inline int2 __attribute__((overloadable)) make_int2(uint2 a) {
//  return make_int2((int)(a.x), (int)(a.y));
//}
//
//static inline int2 __attribute__((overloadable)) make_int2(float2 a) {
//  return make_int2((int)(a.x), (int)(a.y));
//}
//
//static inline uint2 __attribute__((overloadable)) make_uint2(uint s) {
//  return make_uint2(s, s);
//}
//
//static inline uint2 __attribute__((overloadable)) make_uint2(uint3 a) {
//  return make_uint2(a.x, a.y);
//}
//
//static inline uint2 __attribute__((overloadable)) make_uint2(int2 a) {
//  return make_uint2((uint)(a.x), (uint)(a.y));
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(float s) {
//  return make_float3(s, s, s);
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(int s) {
//  return make_float3((float)s, (float)s, (float)s);
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(float2 a) {
//  return make_float3(a.x, a.y, 0.0f);
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(float2 a, float s) {
//  return make_float3(a.x, a.y, s);
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(float4 a) {
//  return make_float3(a.x, a.y, a.z);
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(int3 a) {
//  return make_float3((float)(a.x), (float)(a.y), (float)(a.z));
//}
//
//static inline float3 __attribute__((overloadable)) make_float3(uint3 a) {
//  return make_float3((float)(a.x), (float)(a.y), (float)(a.z));
//}
//
//static inline int3 __attribute__((overloadable)) make_int3(int s) {
//  return make_int3(s, s, s);
//}
//
//static inline int3 __attribute__((overloadable)) make_int3(int2 a) {
//  return make_int3(a.x, a.y, 0);
//}
//
//static inline int3 __attribute__((overloadable)) make_int3(int2 a, int s) {
//  return make_int3(a.x, a.y, s);
//}
//
//static inline int3 __attribute__((overloadable)) make_int3(uint3 a) {
//  return make_int3((int)(a.x), (int)(a.y), (int)(a.z));
//}
//
//static inline int3 __attribute__((overloadable)) make_int3(float3 a) {
//  return make_int3((int)(a.x), (int)(a.y), (int)(a.z));
//}
//
//static inline uint3 __attribute__((overloadable)) make_uint3(uint s) {
//  return make_uint3(s, s, s);
//}
//
//static inline uint3 __attribute__((overloadable)) make_uint3(uint2 a) {
//  return make_uint3(a.x, a.y, 0);
//}
//
//static inline uint3 __attribute__((overloadable)) make_uint3(uint2 a, uint s) {
//  return make_uint3(a.x, a.y, s);
//}
//
//static inline uint3 __attribute__((overloadable)) make_uint3(uint4 a) {
//  return make_uint3(a.x, a.y, a.z);
//}
//
//static inline uint3 __attribute__((overloadable)) make_uint3(int3 a) {
//  return make_uint3((uint)(a.x), (uint)(a.y), (uint)(a.z));
//}
//
//static inline float4 __attribute__((overloadable)) make_float4(float s) {
//  return make_float4(s, s, s, s);
//}
//
//static inline float4 __attribute__((overloadable)) make_float4(int s) {
//  return make_float4((float)s, (float)s, (float)s, (float)s);
//}
//
//
//static inline float4 __attribute__((overloadable)) make_float4(float3 a) {
//  return make_float4(a.x, a.y, a.z, 0.0f);
//}
//
//static inline float4 __attribute__((overloadable)) make_float4(float3 a, float w) {
//  return make_float4(a.x, a.y, a.z, w);
//}
//
//static inline float4 __attribute__((overloadable)) make_float4(int4 a) {
//  return make_float4((float)(a.x), (float)(a.y), (float)(a.z), (float)(a.w));
//}
//
//static inline float4 __attribute__((overloadable)) make_float4(uint4 a) {
//  return make_float4((float)(a.x), (float)(a.y), (float)(a.z), (float)(a.w));
//}
//
//static inline int4 __attribute__((overloadable)) make_int4(int s) {
//  return make_int4(s, s, s, s);
//}
//
//static inline int4 __attribute__((overloadable)) make_int4(int3 a) {
//  return make_int4(a.x, a.y, a.z, 0);
//}
//
//static inline int4 __attribute__((overloadable)) make_int4(int3 a, int w) {
//  return make_int4(a.x, a.y, a.z, w);
//}
//
//static inline int4 __attribute__((overloadable)) make_int4(uint4 a) {
//  return make_int4((int)(a.x), (int)(a.y), (int)(a.z), (int)(a.w));
//}
//
//static inline int4 __attribute__((overloadable)) make_int4(float4 a) {
//  return make_int4((int)(a.x), (int)(a.y), (int)(a.z), (int)(a.w));
//}
//
//
//static inline uint4 __attribute__((overloadable)) make_uint4(uint s) {
//  return make_uint4(s, s, s, s);
//}
//
//static inline uint4 __attribute__((overloadable)) make_uint4(uint3 a) {
//  return make_uint4(a.x, a.y, a.z, 0);
//}
//
//static inline uint4 __attribute__((overloadable)) make_uint4(uint3 a, uint w) {
//  return make_uint4(a.x, a.y, a.z, w);
//}
//
//static inline uint4 __attribute__((overloadable)) make_uint4(int4 a) {
//  return make_uint4((uint)(a.x), (uint)(a.y), (uint)(a.z), (uint)(a.w));
//}

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

#define make_float2(x, y)        ((float2)(x, y))
#define make_float3(x, y, z)     ((float3)(x, y, z))
#define make_float4(x, y, z, w)  ((float4)(x, y, z, w))
#define make_int2(x, y)          ((int2)(x, y))
#define make_int3(x, y, z)       ((int3)(x, y, z))
#define make_int4(x, y, z, w)    ((int4)(x, y, z, w))
#define make_uint2(x, y)         ((uint2)(x, y))
#define make_uint3(x, y, z)      ((uint3)(x, y, z))
#define make_uint4(x, y, z, w)   ((uint4)(x, y, z, w))
#define make_uchar4(x, y, z, w)  ((uchar4)(x, y, z, w))

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
  val.m11=r0.x; val.m12=r0.y; val.m13=r0.z;
  val.m21=r1.x; val.m22=r1.y; val.m23=r1.z;
  val.m31=r2.x; val.m32=r2.y; val.m33=r2.z;
  return val;
}

static inline float3 __attribute__((overloadable)) matrix_mul(float3x3 m, float3 v) {
  return make_float3(
          m.m11*v.x + m.m12*v.y + m.m13*v.z,
          m.m21*v.x + m.m22*v.y + m.m23*v.z,
          m.m31*v.x + m.m32*v.y + m.m33*v.z
  );
}

//
//////////////////////////////////////////////////////////////////////////////////
//// min
//////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fminf(float a, float b) {
  return fmin(a, b);
}

static inline  float2 __attribute__((overloadable)) fminf(float2 a, float2 b) {
  return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

static inline float3 __attribute__((overloadable)) fminf(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

static inline  float4 __attribute__((overloadable)) fminf(float4 a, float4 b) {
  return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

//////////////////////////////////////////////////////////////////////////////////
//// max
//////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fmaxf(float a, float b) {
  return fmax(a, b);
}

static inline float2 __attribute__((overloadable)) fmaxf(float2 a, float2 b) {
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

static inline float3 __attribute__((overloadable)) fmaxf(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

static inline float4 __attribute__((overloadable)) fmaxf(float4 a, float4 b) {
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
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
  return make_float2(floorf(v.x), floorf(v.y));
}

static inline float3 __attribute__((overloadable)) floorf(float3 v) {
  return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

static inline float4 __attribute__((overloadable)) floorf(float4 v) {
  return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fracf(float v) {
  return v - floorf(v);
}

static inline float2 __attribute__((overloadable)) fracf(float2 v) {
  return make_float2(fracf(v.x), fracf(v.y));
}

static inline float3 __attribute__((overloadable)) fracf(float3 v) {
  return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}

static inline float4 __attribute__((overloadable)) fracf(float4 v) {
  return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

static inline float __attribute__((overloadable)) fmodf(float a, float b) {
  return fmod(a,b);
}

static inline float2 __attribute__((overloadable)) fmodf(float2 a, float2 b) {
  return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}

static inline float3 __attribute__((overloadable)) fmodf(float3 a, float3 b) {
  return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}

static inline float4 __attribute__((overloadable)) fmodf(float4 a, float4 b) {
  return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
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
//#define powf pow

#endif
