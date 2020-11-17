//
// Created by denn nevera on 04/06/2020.
//

#pragma once

#include "dehancer/gpu/ocio/GamaParams.h"

namespace dehancer{
    namespace ocio {
        namespace REC709_22 {
            static const ocio::GammaParameters gamma_parameters = {
                    .lin_side_break  = 0.018f,
                    .lin_side_coeff  = 4.5f,
                    .lin_side_offset = 0.099f,
                    .lin_side_slope  = 1.099f,
                    .gama_side_break = 0.081f,
                    .base = 0.45f,
                    .enabled = true
            };
        };

        namespace DEHANCER_MLUT_REC709 {
            static const ocio::GammaParameters gamma_parameters = {
                    .lin_side_break  = 0.0f,
                    .lin_side_coeff  = 1.0f,
                    .lin_side_offset = 0.0f,
                    .lin_side_slope  = 1.0f,
                    .gama_side_break = 0.0f,
                    .base = (1.0f/2.0f)/(1.0f/2.4f),
                    .enabled = true
            };
        };

        namespace DEHANCER_MLUT_FC { // Final Cut gamma correction
            static const ocio::GammaParameters gamma_parameters = {
                    .lin_side_break  = 0.0f,
                    .lin_side_coeff  = 1.0f,
                    .lin_side_offset = 0.0f,
                    .lin_side_slope  = 1.0f,
                    .gama_side_break = 0.0f,
                    .base = 2.0f,
                    .enabled = true
            };
        };

        namespace REC709_24 {
            static const ocio::GammaParameters gamma_parameters = {
                    .lin_side_break  = 0.0031308f,
                    .lin_side_coeff  = 12.92f,
                    .lin_side_offset =  0.055f,
                    .lin_side_slope  = 1.055f,
                    .gama_side_break = 0.04045f,
                    .base = 1.0f/2.4f,
                    .enabled = true
            };
        };
    }
}