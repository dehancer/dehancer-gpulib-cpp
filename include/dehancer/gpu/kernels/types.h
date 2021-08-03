//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_TYPES_H
#define DEHANCER_GPULIB_TYPES_H

#if defined(__CUDA_ARCH__)

#define DEHANCER_GPU_CODE 1

#include "dehancer/gpu/kernels/cuda/cuda_types.h"

#elif defined(__METAL_VERSION__)

#define DEHANCER_GPU_CODE 1

#include "dehancer/gpu/kernels/metal/metal_types.h"

#elif defined(CL_VERSION_1_2)

#define DEHANCER_GPU_CODE 1

#include "dehancer/gpu/kernels/opencl/opencl_types.h"

#else

#define DEHANCER_GPU_CODE 0

/**
 * Dummy auto indentation for CLion
 */

typedef  unsigned int uint;

#define __constant const
#define __constant const
#define __read_only const
#define __write_only
#define __read_write

#define DHCR_BIND_TEXTURE(N)
#define DHCR_BIND_BUFFER(N)

#define DHCR_KERNEL_GID_1D
#define DHCR_KERNEL_GID_2D
#define DHCR_KERNEL_GID_3D

#define DHCR_KERNEL
#define DHCR_DEVICE_FUNC
#define DHCR_DEVICE_ARG
#define DHCR_THREAD_ARG
#define DHCR_CONST_ARG
#define uint_ref_t unsigned int
#define int_ref_t int
#define float_ref_t  float
#define float2_ref_t float2
#define float3_ref_t float3
#define float4_ref_t float4

#define uint2_ref_t uint2
#define uint3_ref_t uint3
#define uint4_ref_t uint4

#define int2_ref_t int2
#define int3_ref_t int3
#define int4_ref_t int4

#define bool_ref_t bool
#define bool_t bool
#define bool2_ref_t bool2
#define bool3_ref_t bool3
#define bool4_ref_t bool4

#include "dehancer/gpu/Typedefs.h"
#include <climits>
#include <cfloat>
#include <cmath>

#endif

#define DHCR_READ_ONLY  __read_only
#define DHCR_WRITE_ONLY __write_only
#define DHCR_READ_WRITE __read_write

typedef enum {
    DHCR_ADDRESS_CLAMP,
    DHCR_ADDRESS_BORDER,
    DHCR_ADDRESS_WRAP
} DHCR_EdgeMode;

typedef enum {
    DHCR_Normal     = 0,
    DHCR_Luminosity = 1,
    DHCR_Color      = 2,
    DHCR_Mix        = 3,
    DHCR_Overlay    = 4,
    DHCR_Min        = 5,
    DHCR_Max        = 6,
    DHCR_Add        = 7,
    DHCR_Subtract   = 8
} DHCR_BlendingMode;

typedef enum {
    DHCR_Bilinear = 0 ,
    DHCR_Bicubic  = 1,
    DHCR_BoxAverage  = 2
} DHCR_InterpolationMode;

typedef struct { //__attribute__ ((packed))
    bool_t  enabled;
    float base;
    float lin_side_break;
    float lin_side_coeff;
    float lin_side_offset;
    float lin_side_slope;
    float gamma_side_break;
} DHCR_GammaParameters;

typedef struct {
    bool_t  enabled;
    float base;
    float log_side_slope;
    float log_side_offset;
    float lin_side_slope;
    float lin_side_offset;
    float lin_side_break;
    float log_side_break;
    float linear_slope;
    float linear_offset;
    float log2_base;
} DHCR_LogParameters ;

typedef struct {
    bool_t   enabled;
    uint   size;
    uint   channels;
#if !DEHANCER_GPU_CODE
    float* data;
#endif
} DHCR_LutParameters;

typedef enum { // __attribute__ ((packed, aligned (16))) {
    DHCR_Forward = 0,
    DHCR_Inverse,
    DHCR_None
} DHCR_TransformDirection;


typedef enum  {
    DHCR_log_linear = 0,
    DHCR_pow_linear
} DHCR_TransformType;

#endif //DEHANCER_GPULIB_TYPES_H
