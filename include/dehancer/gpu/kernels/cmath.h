//
// Created by denn on 15.01.2021.
//

#ifndef DEHANCER_GPULIB_CMATH_H
#define DEHANCER_GPULIB_CMATH_H

#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/constants.h"

#if defined(__CUDA_ARCH__)
#include <cmath>
#include "dehancer/gpu/kernels/cuda/cuda.h"
#include "dehancer/gpu/kernels/cuda/cmath.h"

#elif defined(__METAL_VERSION__)

#include "dehancer/gpu/kernels/metal/cmath.h"

#elif defined(CL_VERSION_1_2)

#include "dehancer/gpu/kernels/opencl/opencl.h"
#include "dehancer/gpu/kernels/opencl/cmath.h"

static inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) powf(float a, float b) {
  return pow(a,b);
}

static inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) log2f(float a) {
  return log2(a);
}

static inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) log10f(float a) {
  return log10(a);
}

#endif

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) to_float2(float C) { return make_float2(C,C); }
static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) to_float3(float C) { return make_float3(C,C,C); }
static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) to_float4(float C) { return make_float4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) to_float2(int C) { return make_float2(C,C); }
static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) to_float3(int C) { return make_float3(C,C,C); }
static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) to_float4(int C) { return make_float4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) to_float2(int2 C) { return make_float2(C.x,C.y); }
static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) to_float3(int3 C) { return make_float3(C.x,C.y,C.z); }
static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) to_float4(int4 C) { return make_float4(C.x,C.y,C.z,C.w); }

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) to_float2(uint C) { return make_float2(C,C); }
static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) to_float3(uint C) { return make_float3(C,C,C); }
static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) to_float4(uint C) { return make_float4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) to_float2(uint2 C) { return make_float2(C.x,C.y); }
static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) to_float3(uint3 C) { return make_float3(C.x,C.y,C.z); }
static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) to_float4(uint4 C) { return make_float4(C.x,C.y,C.z,C.w); }

static inline DHCR_DEVICE_FUNC int2 __attribute__((overloadable)) to_int2(int C) { return make_int2(C,C); }
static inline DHCR_DEVICE_FUNC int3 __attribute__((overloadable)) to_int3(int C) { return make_int3(C,C,C); }
static inline DHCR_DEVICE_FUNC int4 __attribute__((overloadable)) to_int4(int C) { return make_int4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC int2 __attribute__((overloadable)) to_int2(uint C) { return make_int2(C,C); }
static inline DHCR_DEVICE_FUNC int3 __attribute__((overloadable)) to_int3(uint C) { return make_int3(C,C,C); }
static inline DHCR_DEVICE_FUNC int4 __attribute__((overloadable)) to_int4(uint C) { return make_int4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC int2 __attribute__((overloadable)) to_int2(float C) { return make_int2(C,C); }
static inline DHCR_DEVICE_FUNC int3 __attribute__((overloadable)) to_int3(float C) { return make_int3(C,C,C); }
static inline DHCR_DEVICE_FUNC int4 __attribute__((overloadable)) to_int4(float C) { return make_int4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC int2 __attribute__((overloadable)) to_int2(float2 C) { return make_int2(C.x,C.y); }
static inline DHCR_DEVICE_FUNC int3 __attribute__((overloadable)) to_int3(float3 C) { return make_int3(C.x,C.y,C.z); }
static inline DHCR_DEVICE_FUNC int4 __attribute__((overloadable)) to_int4(float4 C) { return make_int4(C.x,C.y,C.z,C.w); }

static inline DHCR_DEVICE_FUNC int2 __attribute__((overloadable)) to_int2(uint2 C) { return make_int2(C.x,C.y); }
static inline DHCR_DEVICE_FUNC int3 __attribute__((overloadable)) to_int3(uint3 C) { return make_int3(C.x,C.y,C.z); }
static inline DHCR_DEVICE_FUNC int4 __attribute__((overloadable)) to_int4(uint4 C) { return make_int4(C.x,C.y,C.z,C.w); }

static inline DHCR_DEVICE_FUNC uint2 __attribute__((overloadable)) to_uint2(uint C) { return make_uint2(C,C); }
static inline DHCR_DEVICE_FUNC uint3 __attribute__((overloadable)) to_uint3(uint C) { return make_uint3(C,C,C); }
static inline DHCR_DEVICE_FUNC uint4 __attribute__((overloadable)) to_uint4(uint C) { return make_uint4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC uint2 __attribute__((overloadable)) to_uint2(int C) { return make_uint2(C,C); }
static inline DHCR_DEVICE_FUNC uint3 __attribute__((overloadable)) to_uint3(int C) { return make_uint3(C,C,C); }
static inline DHCR_DEVICE_FUNC uint4 __attribute__((overloadable)) to_uint4(int C) { return make_uint4(C,C,C,C); }

static inline DHCR_DEVICE_FUNC uint2 __attribute__((overloadable)) to_uint2(float2 C) { return make_uint2(C.x,C.y); }
static inline DHCR_DEVICE_FUNC uint3 __attribute__((overloadable)) to_uint3(float3 C) { return make_uint3(C.x,C.y,C.z); }
static inline DHCR_DEVICE_FUNC uint4 __attribute__((overloadable)) to_uint4(float4 C) { return make_uint4(C.x,C.y,C.z,C.w); }

static inline DHCR_DEVICE_FUNC uint2 __attribute__((overloadable)) to_uint2(int2 C) { return make_uint2(C.x,C.y); }
static inline DHCR_DEVICE_FUNC uint3 __attribute__((overloadable)) to_uint3(int3 C) { return make_uint3(C.x,C.y,C.z); }
static inline DHCR_DEVICE_FUNC uint4 __attribute__((overloadable)) to_uint4(int4 C) { return make_uint4(C.x,C.y,C.z,C.w); }

static inline float4 __attribute__((overloadable)) to_float4(float3 a, float w) {
  return make_float4(a.x, a.y, a.z, w);
}

static inline float3 __attribute__((overloadable)) to_float3(float4 a) {
  return make_float3(a.x, a.y, a.z);
}

static inline DHCR_DEVICE_FUNC float  __attribute__((overloadable)) permute(float x)
{
  return floorf(fmodf(((x*34.0f)+1.0f)*x, 289.0f));
}

static inline DHCR_DEVICE_FUNC float taylor_inv_sqrt(float r)
{
  return 1.79284291400159f - 0.85373472095314f * r;
}

/***
 * fract
 * @param v
 * @return
 */
#if defined(__CUDA_ARCH__) || defined(CL_VERSION_1_2)
static inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) fract(float v) {
  return fracf(v);
}

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) fract(float2 v) {
  return fracf(v);
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) fract(float3 v) {
  return fracf(v);
}

/***
 * abs
 */
static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) abs(float3 v) {
  return make_float3(fabs(v.x),fabs(v.y),fabs(v.z));
}

/***
 * powf
 */
static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) powf(float2 a, float2 b) {
  return make_float2(powf(a.x,b.x),powf(a.y,b.y));
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) powf(float3 a, float3 b) {
  return make_float3(powf(a.x,b.x),powf(a.y,b.y),powf(a.z,b.z));
}

static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) powf(float4 a, float4 b) {
  return make_float4(powf(a.x,b.x),powf(a.y,b.y),powf(a.z,b.z),powf(a.w,b.w));
}

/***
 * log2f
 */
static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) log2f(float2 a) {
  return make_float2(log2f(a.x),log2f(a.y));
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) log2f(float3 a) {
  return make_float3(log2f(a.x),log2f(a.y),log2f(a.z));
}

static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) log2f(float4 a) {
  return make_float4(log2f(a.x),log2f(a.y),log2f(a.z),log2f(a.w));
}

/***
 * log10f
 */
static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) log10f(float2 a) {
  return make_float2(log10f(a.x),log10f(a.y));
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) log10f(float3 a) {
  return make_float3(log10f(a.x),log10f(a.y),log10f(a.z));
}

static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) log10f(float4 a) {
  return make_float4(log10f(a.x),log10f(a.y),log10f(a.z),log10f(a.w));
}

#endif


/***
 * lum
 */
static inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) lum(float3 c) {
  return dot(c, kIMP_Y_YCbCr_factor);
}

/***
 * Linear -> log and log -> linear
 * @param in
 * @param slope
 * @param offset
 * @param direction
 * @param opacity
 * @return
 */
static inline DHCR_DEVICE_FUNC float __attribute__((overloadable))  linear_log(float in, float slope, float offset, DHCR_TransformDirection direction, float opacity) {
  float result = in;
  if (direction == DHCR_None) return result;
  if (slope==0.0f) return result;
  
  if (direction == DHCR_Forward) {
    result = powf( 2.0f, result*slope-offset);
  }
  else {
    result = (log2f(result) + offset) / slope;
  }
  
  result = mix (in, result, smoothstep(0.0f,1.0f, opacity));
  
  return result;
}

static inline DHCR_DEVICE_FUNC float __attribute__((overloadable))  linear_log(float in, float slope, float offset, DHCR_TransformDirection direction) {
  float result = in;
  if (direction == DHCR_None) return result;
  if (slope==0.0f) return result;
  if (direction == DHCR_Forward) {
    result = powf( 2.0f, result*slope-offset);
  }
  else {
    result = (log2f(result) + offset) / slope;
  }
  return result;
}

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable))  linear_log(float2 in, float slope, float offset, DHCR_TransformDirection direction) {
  float2 result = in;
  if (direction == DHCR_None) return result;
  if (slope==0.0f) return result;
  if (direction == DHCR_Forward) {
    result = powf( to_float2(2.0f), result*to_float2(slope)-to_float2(offset));
  }
  else {
    result = (log2f(result) + to_float2(offset)) / to_float2(slope);
  }
  return result;
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable))  linear_log(float3 in, float slope, float offset, DHCR_TransformDirection direction) {
  float3 result = in;
  if (direction == DHCR_None) return result;
  if (slope==0.0f) return result;
  if (direction == DHCR_Forward) {
    result = powf( to_float3(2.0f), result*to_float3(slope)-to_float3(offset));
  }
  else {
    result = (log2f(result) + to_float3(offset)) / to_float3(slope);
  }
  return result;
}

static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) linear_log(float4 in, float slope, float offset, DHCR_TransformDirection direction) {
  float4 result = in;
  if (direction == DHCR_None) return result;
  if (slope==0.0f) return result;
  if (direction == DHCR_Forward) {
    result = powf( to_float4(2.0f), result*to_float4(slope)-to_float4(offset));
  }
  else {
    result = (log2f(result) + to_float4(offset)) / to_float4(slope);
  }
  return result;
}

/***
 * Linear -> power and power -> linear
 * @param in
 * @param slope
 * @param offset
 * @param direction
 * @param opacity
 * @return
 */
static inline DHCR_DEVICE_FUNC float __attribute__((overloadable))  linear_pow(float in, float slope, float offset, DHCR_TransformDirection direction, float opacity) {
  float result = in;

  if (direction == DHCR_None) return result;
  
  if (direction == DHCR_Forward) {
    result = powf( result, slope) ;
  }
  else {
    result =  powf( result, 1.0f/slope) ;
  }
  
  result = mix (in, result, smoothstep(0.0f,1.0f, opacity));
  
  return result;
}

static inline DHCR_DEVICE_FUNC float __attribute__((overloadable))  linear_pow(float in, float slope, float offset, DHCR_TransformDirection direction) {
  float result = in;
  
  if (direction == DHCR_None) return result;
  
  if (direction == DHCR_Forward) {
    result = powf( result, slope) ;
  }
  else {
    result =  powf( result, 1.0f/slope) ;
  }
  
  return result;
}

static inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable))  linear_pow(float2 in, float slope, float offset, DHCR_TransformDirection direction) {
  float2 result = in;
  if (direction == DHCR_None) return result;

  if (direction == DHCR_Forward) {
    result = powf( result, to_float2(slope)) ;
  }
  else {
    result =  powf( result, to_float2(1.0f/slope)) ;
  }

  return result;
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable))  linear_pow(float3 in, float slope, float offset, DHCR_TransformDirection direction) {
  float3 result = in;
  
  if (direction == DHCR_None) return result;
  
  if (direction == DHCR_Forward) {
    result = powf( result, to_float3(slope)) ;
  }
  else {
    result =  powf( result, to_float3(1.0f/slope));
  }
  
  return result;
}

static inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) linear_pow(float4 in, float slope, float offset, DHCR_TransformDirection direction) {
  float4 result = in;
  
  if (direction == DHCR_None) return result;
  
  if (direction == DHCR_Forward) {
    result = powf( result, to_float4(slope));
  }
  else {
    result =  powf( result, to_float4(1.0f/slope));
  }
  
  return result;
}


#endif //DEHANCER_GPULIB_CMATH_H
