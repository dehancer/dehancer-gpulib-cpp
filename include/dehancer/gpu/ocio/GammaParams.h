//
// Created by denn nevera on 04/06/2020.
//

#pragma once


#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/Typedefs.h"

namespace dehancer::ocio {
    using GammaParameters = DHCR_GammaParameters;
    float3 apply_gama_forward(float3 in, GammaParameters params);
    float3 apply_gama_inverse(float3 in, GammaParameters params);
}