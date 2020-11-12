//
// Created by denn nevera on 04/06/2020.
//

#pragma once

#include "Params.h"

namespace dehancer {

    namespace ocio {

        struct GammaParameters {
            float lin_side_break  = 0;
            float lin_side_coeff  = 1;
            float lin_side_offset = 0;
            float lin_side_slope  = 1;
            float gama_side_break = 0;
            float base = 0.45f;
            bool  enabled = false;
        };

        __METAL_INLINE__ float3 apply_gama_forward(float3 in, GammaParameters params);
        __METAL_INLINE__ float3 apply_gama_inverse(float3 in, GammaParameters params);
    }
}