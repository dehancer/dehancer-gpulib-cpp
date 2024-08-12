//
// Created by denn nevera on 2019-10-18.
//

#pragma once

#include "dehancer/gpu/DeviceConfig.h"
#include "dehancer/gpu/ocio/LogParams.h"
#include "dehancer/gpu/ocio/GammaParams.h"
#include "dehancer/gpu/ocio/LutParams.h"

namespace dehancer {

#if !DEHANCER_GPU_CODE
    static inline dehancer::float4 make_float4(float x, float y, float z, float w) {
      return dehancer::float4({x, y, z, w});
    }
#endif
    
#include "dehancer/gpu/kernels/stream_space.h"
    
    using StreamSpace = DHCR_StreamSpace;
    using StreamSpaceDirection = DHCR_TransformDirection;
}