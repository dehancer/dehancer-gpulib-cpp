//
// Created by denn on 10.02.2021.
//

#ifndef DEHANCER_GPULIB_CUDA_TYPES_H
#define DEHANCER_GPULIB_CUDA_TYPES_H


typedef  unsigned int uint;

#define __constant const
#define __constant const
#define __read_only const
#define __write_only
#define __read_write

#define DHCR_BIND_TEXTURE(N)
#define DHCR_BIND_BUFFER(N)

#define DHCR_KERNEL         extern "C" __global__
#define DHCR_DEVICE_FUNC   __device__
#define DHCR_DEVICE_ARG
#define DHCR_THREAD_ARG
#define DHCR_CONST_ARG

#define DHCR_KERNEL_GID_1D
#define DHCR_KERNEL_GID_2D
#define DHCR_KERNEL_GID_3D

#define bool_ref_t bool
#define bool_t bool
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

#define bool2_ref_t uint2
#define bool3_ref_t uint3
#define bool4_ref_t uint4

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

#endif //DEHANCER_GPULIB_CUDA_TYPES_H
