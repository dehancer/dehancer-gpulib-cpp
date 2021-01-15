//
// Created by denn on 15.01.2021.
//

#ifndef DEHANCER_GPULIB_CMATH_H
#define DEHANCER_GPULIB_CMATH_H

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

#endif


inline DHCR_DEVICE_FUNC float  __attribute__((overloadable)) permute(float x)
{
  return floor(fmod(((x*34.0)+1.0)*x, 289.0));
}

inline DHCR_DEVICE_FUNC float taylor_inv_sqrt(float r)
{
  return 1.79284291400159f - 0.85373472095314f * r;
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) fract(float v) {
  return fracf(v);
}

inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) fract(float2 v) {
  return fracf(v);
}

inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) fract(float3 v) {
  return fracf(v);
}

inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) abs(float3 v) {
  return (float3){fabs(v.x),fabs(v.y),fabs(v.z)};
}

inline DHCR_DEVICE_FUNC float2 __attribute__((overloadable)) powf(float2 a, float2 b) {
  return make_float2(powf(a.x,b.x),powf(a.y,b.y));
}

inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) powf(float3 a, float3 b) {
  return make_float3(powf(a.x,b.x),powf(a.y,b.y),powf(a.z,b.z));
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) lum(float3 c) {
  return dot(c, kIMP_Y_YCbCr_factor);
}

#endif //DEHANCER_GPULIB_CMATH_H
