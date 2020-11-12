//
// Created by denn nevera on 04/06/2020.
//

#pragma once

#include "Params.h"

namespace dehancer {

    namespace ocio {

        struct LogParameters {
            float log_side_slope = 0;
            float log_side_offset = 0;
            float lin_side_slope = 0;
            float lin_side_offset = 0;
            float lin_side_break = 0;
            float log_side_break = 0;
            float linear_slope = 0;
            float linear_offset = 0;
            float log2_base = 1;
            float base = 2;
            bool  enabled = false;
        };

        __METAL_INLINE__ float3 apply_log_forward(float3 in, dehancer::ocio::LogParameters params);
        __METAL_INLINE__ float3 apply_log_inverse(float3 in, dehancer::ocio::LogParameters params);
    }
}