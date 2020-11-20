//
// Created by denn nevera on 04/06/2020.
//

#pragma once

#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
using namespace simd;

namespace dehancer {
    namespace ocio {
        inline static float max(float x, float y) { return simd::max(x,y); }
    }
}
#define __METAL_INLINE__ inline static

#else

#include "dehancer/gpu/DeviceConfig.h"
#include "dehancer/math.hpp"
#include <cfloat>

namespace dehancer::ocio {
    inline static float max(float x, float y) { return std::fmax(x,y); }
}

#define __METAL_INLINE__

#endif
