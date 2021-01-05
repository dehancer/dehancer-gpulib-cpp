//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_CONSTANTS_H
#define DEHANCER_GPULIB_CONSTANTS_H

#include "dehancer/gpu/kernels/types.h"

static __constant __DEHANCER_DEVICE_FUNC__ float3 kIMP_Y_YUV_factor = {
    0.2125, 0.7154, 0.0721
};

#endif //DEHANCER_GPULIB_CONSTANTS_H
