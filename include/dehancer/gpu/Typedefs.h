//
// Created by denn nevera on 12/11/2020.
//

#ifndef DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H
#define DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H

#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include "aoBenchKernel.h"

using namespace metal;

#elif CL_VERSION_1_2

#else

#include "dehancer/gpu/kernels/types.h"
#include "dehancer/math.hpp"

namespace dehancer {
    typedef dehancer::math::float2 float2;
    typedef dehancer::math::float3 float3;
    typedef dehancer::math::float4 float4;
    typedef dehancer::math::float2x2 float2x2;
    typedef dehancer::math::float3x3 float3x3;
    typedef dehancer::math::float4x4 float4x4;
}
#endif

#endif //DEHANCER_GPULIB_CPP_GPUTYPEDEFS_H
