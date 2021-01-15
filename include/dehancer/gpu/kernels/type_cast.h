//
// Created by denn on 14.01.2021.
//

#ifndef DEHANCER_VIDEO_TYPE_CAST_H
#define DEHANCER_VIDEO_TYPE_CAST_H

#include "dehancer/gpu/kernels/types.h"

#if defined(__CUDA_ARCH__)

#include <cuda.h>

#elif defined(CL_VERSION_1_2)

inline DHCR_DEVICE_FUNC float uint_as_float(uint m ) {
  return as_float(m);
}

inline DHCR_DEVICE_FUNC uint float_as_uint(float m ) {
  return as_uint(m);
}

#else

typedef union {
    float f; uint u;
} float_bitwise;

inline DHCR_DEVICE_FUNC float uint_as_float(uint m ) {
  float_bitwise v; v.u = m;
  return v.f;
}

inline DHCR_DEVICE_FUNC uint float_as_uint(float m ) {
  float_bitwise v; v.f = m;
  return v.u;
}

#endif

inline DHCR_DEVICE_FUNC uint2 float2_as_uint2(float2 m ) {
  return make_uint2(float_as_uint(m.x), float_as_uint(m.y));
}

inline DHCR_DEVICE_FUNC uint3 float3_as_uint3(float3 m ) {
  return make_uint3(float_as_uint(m.x), float_as_uint(m.y), float_as_uint(m.z));
}

inline DHCR_DEVICE_FUNC uint4 float4_as_uint4(float4 m ) {
  return make_uint4(float_as_uint(m.x), float_as_uint(m.y), float_as_uint(m.z), float_as_uint(m.w));
}

__constant uint ieeeMantissa = 0x007FFFFF; // binary32 mantissa bitmask
__constant uint ieeeOne      = 0x3F800000; // 1.0 in IEEE binary32

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
static inline DHCR_DEVICE_FUNC float float_construct( uint m ) {
  
  m &= ieeeMantissa;          // Keep only mantissa bits (fractional part)
  m |= ieeeOne;               // Add fractional part to 1.0
  
  float  f = uint_as_float(m);  // Range [1:2]
  return f - 1.0f;              // Range [0:1]
}

#endif //DEHANCER_VIDEO_TYPE_CAST_H
