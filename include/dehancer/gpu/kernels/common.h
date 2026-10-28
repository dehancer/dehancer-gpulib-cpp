#ifndef DEHANCER_GPULIB_COMMON_LIB_H
#define DEHANCER_GPULIB_COMMON_LIB_H

#if defined(__CUDA_ARCH__)

#include "dehancer/gpu/kernels/cuda/cuda.h"

#elif defined(__METAL_VERSION__)

#include "dehancer/gpu/kernels/metal/metal.h"

#elif defined(__OPENCL_C_VERSION__)

#include "dehancer/gpu/kernels/opencl/opencl.h"

#endif

#include "dehancer/gpu/kernels/types.h"

#endif  // DEHANCER_GPULIB_COMMON_LIB_H
