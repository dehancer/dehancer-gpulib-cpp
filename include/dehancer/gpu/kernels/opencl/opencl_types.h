//
// Created by denn on 10.02.2021.
//

#ifndef DEHANCER_GPULIB_OPENCL_TYPES_H
#define DEHANCER_GPULIB_OPENCL_TYPES_H

#define DHCR_BIND_TEXTURE(N)
#define DHCR_BIND_BUFFER(N)

#define DHCR_KERNEL __kernel
#define DHCR_DEVICE_FUNC
#define DHCR_HOST_DEVICE_FUNC
#define DHCR_DEVICE_ARG __global
#define DHCR_THREAD_ARG
#define DHCR_CONST_ARG
#define DHCR_CONST_ARG_REF(T) DHCR_CONST_ARG T
#define DHCR_BLOCK_MEMORY  local

#define DHCR_KERNEL_GID_1D
#define DHCR_KERNEL_GID_2D
#define DHCR_KERNEL_GID_3D

#define atomic_int  int
#define atomic_bool uint
#define atomic_uint uint

#define bool_ref_t uint
#define bool_t uint
#define uint8_t unsigned char
#define int8_t char

#define uint_ref_t unsigned int
#define int_ref_t int
#define float_ref_t  float
#define float2_ref_t float2
#define float3_ref_t float3
#define float4_ref_t float4
#define float3x3_ref_t float3x3
#define float4x4_ref_t float4x4

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

typedef struct {
    int width;
    int height;
    int depth;
} Size;


typedef struct {
    Size   grid;
    Size   block;
    int    threads_in_grid;
} ComputeSize;

#define texture1d_read_t DHCR_READ_ONLY image1d_t
#define texture1d_write_t DHCR_WRITE_ONLY image1d_t

#define texture2d_read_t DHCR_READ_ONLY image2d_t
#define texture2d_write_t DHCR_WRITE_ONLY image2d_t

#define texture3d_read_t DHCR_READ_ONLY image3d_t
#define texture3d_write_t DHCR_WRITE_ONLY image3d_t

#define float2x2 float4
#define float4x4 float16

typedef union {
    struct {
        float m11; float m12; float m13;
        float m21; float m22; float m23;
        float m31; float m32; float m33;
    };
    struct {
        float3 s1;
        float3 s2;
        float3 s3;
    };
    float3 v[3];
    float entries[9];
    float entries2[3][3];
} float3x3;


#endif //DEHANCER_GPULIB_OPENCL_TYPES_H
