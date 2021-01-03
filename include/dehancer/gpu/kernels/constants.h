//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_CONSTANTS_H
#define DEHANCER_GPULIB_CONSTANTS_H

#if defined(__CUDA_ARCH__)
#define __constant const
#endif

static __constant float3 kIMP_Y_YUV_factor = {0.2125, 0.7154, 0.0721};

#endif //DEHANCER_GPULIB_CONSTANTS_H
