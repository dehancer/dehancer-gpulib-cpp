//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_OPENCL_H
#define DEHANCER_GPULIB_OPENCL_H

#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#include "dehancer/gpu/kernels/constants.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/cmath.h"
#if DEHANCER_GPU_CODE
#include "dehancer/gpu/kernels/opencl/common.h"
#include "dehancer/gpu/kernels/opencl/std_kernels.h"
#include "dehancer/gpu/kernels/opencl/blur_kernels.h"
#endif

#endif //DEHANCER_GPULIB_OPENCL_H
