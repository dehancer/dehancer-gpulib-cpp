//
// Created by denn nevera on 03/06/2020.
//

#pragma once

#include "dehancer/gpu/ocio/LogParams.h"

namespace dehancer {
    namespace ocio {
        namespace ACEScct {
            const ocio::LogParameters log_parameters = {
                    0.057077625570776259f,
                    0.5547945205479452f,
                    1.0f,
                    0.0f,
                    0.0078125f,
                    0.155251175f,
                    10.5402374f,
                    0.0729055702f,
                    1.0f,
                    2.0f,
                    true
            };
        };
        namespace ACESAP1_DEH2020 {
            static const std::vector<float> forward_ccm_matrix_bfd
                    = {1.705050993f, -0.621792121f, -0.083258872f, 0.000000000f,
                       -0.130256418f, 1.140804737f, -0.010548319f, 0.000000000f,
                       -0.024003357f, -0.128968976f, 1.152972333f, 0.000000000f,
                       0.000000000f, 0.000000000f, 0.000000000f, 1.000000000f};

            static const std::vector<float> inverse_ccm_matrix_bfd
                    = {0.613097402f, 0.339523146f, 0.047379451f, 0.000000000f,
                       0.070193722f, 0.916353879f, 0.013452398f, 0.000000000f,
                       0.020615593f, 0.109569773f, 0.869814634f, 0.000000000f,
                       0.000000000f, 0.000000000f, 0.000000000f, 1.000000000f};
        }
    }
}

