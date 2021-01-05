//
// Created by denn on 05.01.2021.
//

#ifndef DEHANCER_GPULIB_LIB_H
#define DEHANCER_GPULIB_LIB_H

#if defined(__CUDA_ARCH__)
#include "dehancer/gpu/kernels/cuda/cuda.h"
#elif defined(CL_VERSION_1_2)
#include "dehancer/gpu/kernels/opencl/opencl.h"
#endif

#endif // DEHANCER_GPULIB_LIB_H
