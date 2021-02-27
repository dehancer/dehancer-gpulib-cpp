//
// Created by denn nevera on 04/06/2020.
//

#pragma once

#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/Typedefs.h"

namespace dehancer::ocio {
    using LogParameters = DHCR_LogParameters;
    float3 apply_log_forward(float3 in, dehancer::ocio::LogParameters params);
    float3 apply_log_inverse(float3 in, dehancer::ocio::LogParameters params);
}