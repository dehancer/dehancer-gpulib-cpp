//
// Created by denn on 10.02.2021.
//

#pragma once

#include <metal_stdlib>

#define __constant constant
#define __read_only
#define __write_only
#define __read_write

#define DHCR_BIND_TEXTURE(N) [[texture(N)]]
#define DHCR_BIND_BUFFER(N)  [[buffer (N)]]

#define DHCR_KERNEL        kernel
#define DHCR_DEVICE_FUNC
#define DHCR_HOST_DEVICE_FUNC
#define DHCR_DEVICE_ARG    device
#define DHCR_THREAD_ARG    thread
#define DHCR_CONST_ARG     constant
#define DHCR_CONST_ARG_REF(T) DHCR_CONST_ARG T&
#define DHCR_BLOCK_MEMORY  threadgroup


#define DHCR_KERNEL_GID_1D  ,uint __dehancer_kernel_gid_1d__ [[thread_position_in_grid]] \
                            ,uint __dehancer_compute_size_1d__ [[threads_per_grid]]      \
                            ,uint __dehancer_grid_size_1d__    [[threadgroups_per_grid]] \
                            ,uint __dehancer_block_size_1d__   [[threads_per_threadgroup]] \
                            ,uint __dehancer_block_id_1d__     [[threadgroup_position_in_grid]] \
                            ,uint __dehancer_thread_in_block_id_1d__ [[thread_position_in_threadgroup]]
                            
#define DHCR_KERNEL_GID_2D  ,uint2 __dehancer_kernel_gid_2d__ [[thread_position_in_grid]] \
                            ,uint2 __dehancer_compute_size_2d__ [[threads_per_grid]]      \
                            ,uint2 __dehancer_grid_size_2d__    [[threadgroups_per_grid]] \
                            ,uint2 __dehancer_block_size_2d__   [[threads_per_threadgroup]] \
                            ,uint2 __dehancer_block_id_2d__     [[threadgroup_position_in_grid]] \
                            ,uint2 __dehancer_thread_in_block_id_2d__ [[thread_position_in_threadgroup]]

#define DHCR_KERNEL_GID_3D  ,uint3 __dehancer_kernel_gid_3d__ [[thread_position_in_grid]] \
                            ,uint3 __dehancer_compute_size_3d__ [[threads_per_grid]]      \
                            ,uint3 __dehancer_grid_size_3d__    [[threadgroups_per_grid]] \
                            ,uint3 __dehancer_block_size_3d__   [[threads_per_threadgroup]] \
                            ,uint3 __dehancer_block_id_3d__     [[threadgroup_position_in_grid]] \
                            ,uint3 __dehancer_thread_in_block_id_3d__ [[thread_position_in_threadgroup]]

using atomic_int_t = metal::atomic_int;
using atomic_bool_t = metal::atomic_uint;
using atomic_uint_t = metal::atomic_uint;

#define bool_ref_t bool&
#define bool_t bool
#define uint_ref_t uint&
#define int_ref_t int&
#define float_ref_t  float&
#define float2_ref_t float2&
#define float3_ref_t float3&
#define float4_ref_t float4&
#define float3x3_ref_t float3x3&
#define float4x4_ref_t float4x4&

#define uint2_ref_t uint2&
#define uint3_ref_t uint3&
#define uint4_ref_t uint4&

#define int2_ref_t int2&
#define int3_ref_t int3&
#define int4_ref_t int4&

#define bool2_ref_t uint2&
#define bool3_ref_t uint3&
#define bool4_ref_t uint4&

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

#define texture1d_read_t  metal::texture1d<float, metal::access::sample>
#define texture1d_write_t metal::texture1d<float, metal::access::write>

#define texture2d_read_t  metal::texture2d<float, metal::access::sample>
#define texture2d_write_t metal::texture2d<float, metal::access::write>

#define texture3d_read_t  metal::texture3d<float, metal::access::sample>
#define texture3d_write_t metal::texture3d<float, metal::access::write>