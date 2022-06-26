//
// Created by denn on 22.06.2022.
//

#ifndef DEHANCER_GPULIB_HISTOGRAM_KERNEL_H
#define DEHANCER_GPULIB_HISTOGRAM_KERNEL_H

#if defined(__CUDA_ARCH__)

//#include "dehancer/gpu/kernels/cuda/cuda_types.h"

#elif defined(__METAL_VERSION__)

#include "dehancer/gpu/kernels/metal/histogram_image_kernel.h"

#elif defined(CL_VERSION_1_2)

#include "dehancer/gpu/kernels/opencl/histogram_image_kernel.h"

#endif

#endif //DEHANCER_GPULIB_HISTOGRAM_KERNEL_H
