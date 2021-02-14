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
#elif defined(CL_VERSION_1_2)

#include "dehancer/gpu/kernels/opencl/opencl.h"
#include "dehancer/gpu/kernels/opencl/cmath.h"

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) powf(float a, float b) {
  return pow(a,b);
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) log2f(float a) {
  return log2(a);
}

#endif

inline DHCR_DEVICE_FUNC float  __attribute__((overloadable)) permute(float x)
{
  return floor(fmod(((x*34.0)+1.0)*x, 289.0));
}

inline DHCR_DEVICE_FUNC float taylor_inv_sqrt(float r)
{
  return 1.79284291400159f - 0.85373472095314f * r;
}

/***
 * fract
 * @param v
 * @return
 */
inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) fract(float v) {
  return fracf(v);
}

inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) fract(float2 v) {
  return fracf(v);
}

inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) fract(float3 v) {
  return fracf(v);
}

/***
 * abs
 */
inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) abs(float3 v) {
  return (float3){fabs(v.x),fabs(v.y),fabs(v.z)};
}

/***
 * powf
 */
inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) powf(float2 a, float2 b) {
  return make_float2(powf(a.x,b.x),powf(a.y,b.y));
}

inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) powf(float3 a, float3 b) {
return make_float3(powf(a.x,b.x),powf(a.y,b.y),powf(a.z,b.z));
}

inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) powf(float4 a, float4 b) {
  return make_float4(powf(a.x,b.x),powf(a.y,b.y),powf(a.z,b.z),powf(a.w,b.w));
}

/***
 * log2f
 */
inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) log2f(float2 a) {
  return make_float2(log2f(a.x),log2f(a.y));
}

inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) log2f(float3 a) {
  return make_float3(log2f(a.x),log2f(a.y),log2f(a.z));
}

inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) log2f(float4 a) {
  return make_float4(log2f(a.x),log2f(a.y),log2f(a.z),log2f(a.w));
}

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
    result = powf( make_float2(2.0f), result*make_float2(slope)-make_float2(offset));
  }
  else {
    result = (log2f(result) + make_float2(offset)) / make_float2(slope);
  }
  return result;
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable))  linear_log(float3 in, float slope, float offset, DHCR_TransformDirection direction) {
  float3 result = in;
  if (direction == DHCR_None) return result;
  if (slope==0.0f) return result;
  if (direction == DHCR_Forward) {
    result = powf( make_float3(2.0f), result*make_float3(slope)-make_float3(offset));
  }
  else {
    result = (log2f(result) + make_float3(offset)) / make_float3(slope);
  }
  return result;
}

inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) linear_log(float4 in, float slope, float offset, DHCR_TransformDirection direction) {
  float4 result = in;
  if (direction == DHCR_None) return result;
  if (slope==0.0f) return result;
  if (direction == DHCR_Forward) {
    result = powf( make_float4(2.0f), result*make_float4(slope)-make_float4(offset));
  }
  else {
    result = (log2f(result) + make_float4(offset)) / make_float4(slope);
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
    result = powf( result, make_float2(slope)) ;
  }
  else {
    result =  powf( result, make_float2(1.0f/slope)) ;
  }

  return result;
}

static inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable))  linear_pow(float3 in, float slope, float offset, DHCR_TransformDirection direction) {
  float3 result = in;
  
  if (direction == DHCR_None) return result;
  
  if (direction == DHCR_Forward) {
    result = powf( result, make_float3(slope)) ;
  }
  else {
    result =  powf( result, make_float3(1.0f/slope));
  }
  
  return result;
}

inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) linear_pow(float4 in, float slope, float offset, DHCR_TransformDirection direction) {
  float4 result = in;
  
  if (direction == DHCR_None) return result;
  
  if (direction == DHCR_Forward) {
    result = powf( result, make_float4(slope));
  }
  else {
    result =  powf( result, make_float4(1.0f/slope));
  }
  
  return result;
}

#endif //DEHANCER_GPULIB_CMATH_H
