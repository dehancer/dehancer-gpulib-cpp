//
// Created by denn nevera on 12/11/2020.
//

#ifndef DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H
#define DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H

#include "dehancer/math.hpp"

typedef dehancer::math::float2   float2;
typedef dehancer::math::float3   float3;
typedef dehancer::math::float4   float4;
typedef dehancer::math::float2x2 float2x2;
typedef dehancer::math::float3x3 float3x3;
typedef dehancer::math::float4x4 float4x4;

//#if defined(DEHANCER_GPU_OPENCL)
//#define constant const
//#endif

#endif //DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H
