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

#define DHCR_BIND_TEXTURE(N)
#define DHCR_BIND_BUFFER(N)

#define DHCR_KERNEL         extern "C" __global__
#define DHCR_DEVICE_FUNC   __device__
#define DHCR_DEVICE_ARG
#define DHCR_THREAD_ARG
#define DHCR_CONST_ARG
#define int_ref_t int
#define float_ref_t  float
#define float2_ref_t float2
#define float3_ref_t float3
#define float4_ref_t float4

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

typedef enum:int {
    DHCR_ADDRESS_CLAMP,
    DHCR_ADDRESS_BORDER,
    DHCR_ADDRESS_WRAP
} DHCR_EdgeAddress;

#elif defined(CL_VERSION_1_2)

#define DHCR_BIND_TEXTURE(N)
#define DHCR_BIND_BUFFER(N)

#define DHCR_KERNEL __kernel
#define DHCR_DEVICE_FUNC
#define DHCR_DEVICE_ARG __global
#define DHCR_THREAD_ARG
#define DHCR_CONST_ARG
#define int_ref_t int
#define float_ref_t  float
#define float2_ref_t float2
#define float3_ref_t float3
#define float4_ref_t float4

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

typedef enum {
    DHCR_ADDRESS_CLAMP,
    DHCR_ADDRESS_BORDER,
    DHCR_ADDRESS_WRAP
} DHCR_EdgeAddress;

#else

/**
 * Dummy auto indentation for CLion
 */

#define __constant const
#define __constant const
#define __read_only const
#define __write_only
#define __read_write

#define DHCR_BIND_TEXTURE(N)
#define DHCR_BIND_BUFFER(N)

#define DHCR_KERNEL
#define DHCR_DEVICE_FUNC
#define DHCR_DEVICE_ARG
#define DHCR_THREAD_ARG
#define DHCR_CONST_ARG
#define int_ref_t int
#define float_ref_t  float
#define float2_ref_t float2
#define float3_ref_t float3
#define float4_ref_t float4

typedef enum:int {
    DHCR_ADDRESS_CLAMP,
    DHCR_ADDRESS_BORDER,
    DHCR_ADDRESS_WRAP
} DHCR_EdgeAddress;

#endif

#define DHCR_READ_ONLY  __read_only
#define DHCR_WRITE_ONLY __write_only
#define DHCR_READ_WRITE __read_write

#define texture1d_read_t DHCR_READ_ONLY image1d_t
#define texture1d_write_t DHCR_WRITE_ONLY image1d_t

#define texture2d_read_t DHCR_READ_ONLY image2d_t
#define texture2d_write_t DHCR_WRITE_ONLY image2d_t

#define texture3d_read_t DHCR_READ_ONLY image3d_t
#define texture3d_write_t DHCR_WRITE_ONLY image3d_t

#endif //DEHANCER_GPULIB_TYPES_H