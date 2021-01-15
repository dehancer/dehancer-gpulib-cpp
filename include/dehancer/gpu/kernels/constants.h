//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_CONSTANTS_H
#define DEHANCER_GPULIB_CONSTANTS_H

#include "dehancer/gpu/kernels/types.h"

// sRGB luminance(Y) values
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_YUV_factor = {
    0.2125, 0.7154, 0.0721
};

// YCbCr luminance(Y) values
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_YCbCr_factor = {0.299, 0.587, 0.114};

// average
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_mean_factor = {0.3333, 0.3333, 0.3333};

#endif //DEHANCER_GPULIB_CONSTANTS_H
