//
// Created by denn on 05.01.2021.
//

#ifndef DEHANCER_GPULIB_LIB_H
#define DEHANCER_GPULIB_LIB_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/blend_kernels.h"
#include "dehancer/gpu/kernels/overlay_kernels.h"
#if defined(__CUDA_ARCH__)
#include "dehancer/gpu/kernels/unary_kernels_cuda.h"
#else
#include "dehancer/gpu/kernels/unary_kernels.h"
#endif
#include "dehancer/gpu/kernels/resample_kernels.h"
#include "dehancer/gpu/kernels/resize_kernels.h"
#include "dehancer/gpu/kernels/gamma_kernels.h"
#include "dehancer/gpu/kernels/morph_kernels.h"
#include "dehancer/gpu/kernels/channel_utils.h"
#include "dehancer/gpu/kernels/clut_kernels.h"
#include "dehancer/gpu/kernels/stream_kernels.h"
#include "dehancer/gpu/kernels/histogram_kernel.h"


#endif // DEHANCER_GPULIB_LIB_H
