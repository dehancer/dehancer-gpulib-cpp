//
// Created by denn nevera on 04/06/2020.
//

#pragma once

#include "dehancer/gpu/ocio/GammaParams.h"

namespace dehancer::ocio {
    
    namespace REC709_22 {
        static const DHCR_GammaParameters gamma_parameters = {
                .enabled = true,
                .base = 0.45f,
                .lin_side_break  = 0.018f,
                .lin_side_coeff  = 4.5f,
                .lin_side_offset = 0.099f,
                .lin_side_slope  = 1.099f,
                .gamma_side_break = 0.081f
        };
    };
    
    namespace DEHANCER_MLUT_REC709 {
        static const DHCR_GammaParameters gamma_parameters = {
                .enabled = true,
                .base = (1.0f/2.0f)/(1.0f/2.4f),
                .lin_side_break  = 0.0f,
                .lin_side_coeff  = 1.0f,
                .lin_side_offset = 0.0f,
                .lin_side_slope  = 1.0f,
                .gamma_side_break = 0.0f
        };
    };
    
    namespace DEHANCER_MLUT_FC { // Final Cut gamma correction
        static const DHCR_GammaParameters gamma_parameters = {
                .enabled = true,
                .base = 2.0f,
                .lin_side_break  = 0.0f,
                .lin_side_coeff  = 1.0f,
                .lin_side_offset = 0.0f,
                .lin_side_slope  = 1.0f,
                .gamma_side_break = 0.0f
        };
    };
    
    namespace REC709_24 {
        static const DHCR_GammaParameters gamma_parameters = {
                .enabled = true,
                .base = 1.0f/2.4f,
                .lin_side_break  = 0.0031308f,
                .lin_side_coeff  = 12.92f,
                .lin_side_offset =  0.055f,
                .lin_side_slope  = 1.055f,
                .gamma_side_break = 0.04045f
        };
    };
}
