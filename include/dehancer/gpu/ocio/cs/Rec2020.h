//
// Created by denn nevera on 04/06/2020.
//

#pragma once

#include "dehancer/gpu/ocio/GamaParams.h"

namespace dehancer{
    namespace ocio {

        namespace REC2020_22 {
            static const ocio::GammaParameters gamma_parameters = {
                    .lin_side_break  = 0.0181f,
                    .lin_side_coeff  = 4.5f,
                    .lin_side_offset = 0.0993f,
                    .lin_side_slope  = 1.0993f,
                    .gama_side_break = 0.081f,
                    .base = 0.45f,
                    .enabled = true
            };
        }

        namespace REC2020_24 {
            static const ocio::GammaParameters gamma_parameters = {
                    .lin_side_break  = 0.0031308f,
                    .lin_side_coeff  = 12.921f,
                    .lin_side_offset = 0.0553f,
                    .lin_side_slope  = 1.0553f,
                    .gama_side_break = 0.04045f,
                    .base = 1.0f/2.4f,
                    .enabled = true
            };
        }
    }
}