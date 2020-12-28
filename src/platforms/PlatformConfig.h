//
// Created by denn nevera on 15/11/2020.
//

#ifndef DEHANCER_GPULIB_CPP_PLATFORMCONFIG_H
#define DEHANCER_GPULIB_CPP_PLATFORMCONFIG_H

#include "dehancer/gpu/DeviceConfig.h"

#if defined(DEHANCER_GPU_METAL)
#define DEHANCER_GPU_PLATFORM metal
#elif defined(DEHANCER_GPU_CUDA)
//#define DEHANCER_GPU_PLATFORM cuda
#elif defined(DEHANCER_GPU_OPENCL)
#define DEHANCER_GPU_PLATFORM opencl
#endif

#endif
