//
// Created by denn on 14.01.2021.
//

#ifndef DEHANCER_VIDEO_HASH_UTILS_H
#define DEHANCER_VIDEO_HASH_UTILS_H

#include "dehancer/gpu/kernels/type_cast.h"


// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
inline DHCR_DEVICE_FUNC uint __attribute__((overloadable)) hash( uint x ) {
  x += ( x << 10 );
  x ^= ( x >>  6 );
  x += ( x <<  3 );
  x ^= ( x >> 11 );
  x += ( x << 15 );
  return x;
}

// Compound versions of the hashing algorithm I whipped together.
inline DHCR_DEVICE_FUNC uint __attribute__((overloadable)) hash( uint2 v ) { return hash( v.x ^ hash(v.y)                         ); }
inline DHCR_DEVICE_FUNC uint __attribute__((overloadable)) hash( uint3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
inline DHCR_DEVICE_FUNC uint __attribute__((overloadable)) hash( uint4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Pseudo-random value in half-open range [0:1].
inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) random( float   x ) { return float_construct(hash(
          float_as_uint(x))); }
inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) random( float2  v ) { return float_construct(hash(
          float2_as_uint2(v))); }
inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) random( float3  v ) { return float_construct(hash(
          float3_as_uint3(v))); }
inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) random( float4  v ) { return float_construct(hash(
          float4_as_uint4(v))); }

#endif //DEHANCER_VIDEO_HASH_UTILS_H
