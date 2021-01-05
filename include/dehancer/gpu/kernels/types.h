//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_TYPES_H
#define DEHANCER_GPULIB_TYPES_H

#if defined(__CUDA_ARCH__)
#define __constant const
#define __constant const
#define __read_only const
#define __write_only
#define __read_write

#define BIND_TEXTURE(N)
#define BIND_BUFFER(N)

#define __DEHANCER_KERNEL__  extern "C" __global__
#define __DEHANCER_DEVICE_FUNC__ __device__
#define __DEHANCER_DEVICE_ARG__
#define __DEHANCER_THREAD_ARG__
#define __DEHANCER_CONST_ARG__
#define __int_ref int
#define __float_ref  float
#define __float2_ref float2
#define __float3_ref float3
#define __float4_ref float4

#elif defined(CL_VERSION_1_2)

#define BIND_TEXTURE(N)
#define BIND_BUFFER(N)

#define __DEHANCER_KERNEL__ __kernel
#define __DEHANCER_DEVICE_FUNC__
#define __DEHANCER_DEVICE_ARG__ __global
#define __DEHANCER_THREAD_ARG__
#define __DEHANCER_CONST_ARG__
#define __int_ref int
#define __float_ref  float
#define __float2_ref float2
#define __float3_ref float3
#define __float4_ref float4

#else

/**
 * Dummy auto indentation for CLion
 */
#define BIND_TEXTURE(N)
#define BIND_BUFFER(N)

#endif

typedef struct  {
    int gid;
    int size;
} Texel1d;

typedef struct  {
    int2 gid;
    int2 size;
} Texel2d;

typedef struct  {
    int3 gid;
    int3 size;
} Texel3d;

#endif //DEHANCER_GPULIB_TYPES_H
