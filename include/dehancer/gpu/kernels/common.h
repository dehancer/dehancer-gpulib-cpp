//
// Created by denn on 19.01.2021.
//

#ifndef DEHANCER_GPULIB_COMMON_LIB_H
#define DEHANCER_GPULIB_COMMON_LIB_H

#if defined(__CUDA_ARCH__)

#include "dehancer/gpu/kernels/cuda/cuda.h"

#elif defined(__METAL_VERSION__)

#include "dehancer/gpu/kernels/metal/metal.h"

#elif defined(CL_VERSION_1_2)

#include "dehancer/gpu/kernels/opencl/opencl.h"

#endif

#include "dehancer/gpu/kernels/types.h"

#endif //DEHANCER_GPULIB_COMMON_LIB_H
