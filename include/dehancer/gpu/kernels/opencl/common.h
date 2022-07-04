//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_OPENCL_COMMON_H
#define DEHANCER_GPULIB_OPENCL_COMMON_H

#include "dehancer/gpu/kernels/constants.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/type_cast.h"
#include "dehancer/gpu/kernels/hash_utils.h"
#include "dehancer/gpu/kernels/cmath.h"

__constant sampler_t linear_normalized_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define texture1d_read_t DHCR_READ_ONLY image1d_t
#define texture1d_write_t DHCR_WRITE_ONLY image1d_t

#define texture2d_read_t DHCR_READ_ONLY image2d_t
#define texture2d_write_t DHCR_WRITE_ONLY image2d_t

#define texture3d_read_t DHCR_READ_ONLY image3d_t
#define texture3d_write_t DHCR_WRITE_ONLY image3d_t

/**
 * Compute grid info
 */
#define  get_num_blocks_1d() ((int)get_num_groups(0))
#define  get_num_blocks_2d() ((int)get_num_groups(0))

#define  get_block_id1d() ((int)get_group_id(0))
#define  get_block_id2d() ((int2){get_group_id(0), get_group_id(1)})
#define  get_block_id3d() ((int3){get_group_id(0), get_group_id(1), get_group_id(2)})

#define  get_block_size1d() ((int)get_local_size(0))
#define  get_block_size2d() ((int2){get_local_size(0), get_local_size(1)})
#define  get_block_size3d() ((int3){get_local_size(0), get_local_size(1), get_local_size(2)})

#define  get_thread_in_block_id1d() ((int)get_local_id(0))
#define  get_thread_in_block_id2d() ((int2){get_local_id(0), get_local_id(1)})
#define  get_thread_in_block_id3d() ((int3){get_local_id(0), get_local_id(1), get_local_id(2)})

#define  get_thread_in_grid_id1d() ((int)get_global_id(0))
#define  get_thread_in_grid_id2d() ((int2){get_global_id(0), get_global_id(1)})
#define  get_thread_in_grid_id3d() ((int3){get_global_id(0), get_global_id(1), get_global_id(2)})

#define dhr_atomic_fetch_inc(v) atom_inc(&(v))
#define dhr_atomic_store(v,c)   {(v) = (c);}
#define dhr_atomic_load(v)      (v)

#define block_barrier()  barrier(CLK_LOCAL_MEM_FENCE)

/**
 * Kernel computation info
 */

#define get_kernel_texel1d(destination, tex) { \
  tex.gid =  (int)get_global_id(0); \
  tex.size = (int)get_image_width(destination); \
}


#define  get_kernel_tid1d(tid) { \
  tid = (int)get_global_id(0);\
}

#define  get_kernel_tid2d(tid) { \
  tid = (int2){get_global_id(0), get_global_id(1)};\
}

#define  get_kernel_tid3d(tid) { \
  tid = (int3){get_global_id(0), get_global_id(1), get_global_id(2)};  \
}

#define get_kernel_texel2d(destination, tex) { \
  tex.gid =  (int2){get_global_id(0), get_global_id(1)}; \
  tex.size = (int2){get_image_width(destination), get_image_height(destination)}; \
}

#define get_kernel_texel3d(destination, tex) { \
  tex.gid =  (int3){get_global_id(0), get_global_id(1), get_global_id(2)}; \
  tex.size = (int3){get_image_width(destination), get_image_height(destination), get_image_depth(destination)}; \
}

#define get_texture_width(image) get_image_width(image)
#define get_texture_height(image) get_image_height(image)
#define get_texture_depth(image) get_image_depth(image)

static inline  bool __attribute__((overloadable)) get_texel_boundary(Texel1d tex) {
  if (tex.gid >= tex.size) {
    return false;
  }
  return true;
}

static inline  bool __attribute__((overloadable)) get_texel_boundary(Texel2d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y) {
    return false;
  }
  return true;
}

static inline  bool __attribute__((overloadable)) get_texel_boundary(Texel3d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y || tex.gid.z >= tex.size.z) {
    return false;
  }
  return true;
}

static inline  float __attribute__((overloadable)) get_texel_coords(Texel1d tex) {
  return (float)tex.gid / (float)(tex.size - 1);
}

static inline  float2 __attribute__((overloadable)) get_texel_coords(Texel2d tex) {
  return (float2){(float)tex.gid.x / (float)(tex.size.x - 1),
                  (float)tex.gid.y / (float)(tex.size.y - 1)};
}

static inline  float3 __attribute__((overloadable)) get_texel_coords(Texel3d tex) {
  return (float3){
          (float)tex.gid.x / (float)(tex.size.x - 1),
          (float)tex.gid.y / (float)(tex.size.y - 1),
          (float)tex.gid.z / (float)(tex.size.z - 1),
  };
}

// 1D
static inline float4 __attribute__((overloadable)) read_image(__read_only image1d_t source, int gid) {
  return read_imagef(source, nearest_sampler, gid);
}

static inline float4 __attribute__((overloadable)) read_image(__read_only image1d_t source, float coords) {
  return read_imagef(source, linear_normalized_sampler, coords);
}

static inline float4 __attribute__((overloadable)) read_image(__read_only image1d_t source, float4 coords) {
  float4 color = coords;
  color.x = read_imagef(source, linear_normalized_sampler, color.x).x;
  color.y = read_imagef(source, linear_normalized_sampler, color.y).y;
  color.z = read_imagef(source, linear_normalized_sampler, color.z).z;
  return color;
}

static inline void __attribute__((overloadable)) write_image(__write_only image1d_t destination, float4 color, int gid) {
  write_imagef(destination, gid, color);
}


// 2D
static inline float4 __attribute__((overloadable)) read_image(__read_only image2d_t source, int2 gid) {
  int2 coord = (int2)gid;
  int x = get_texture_width(source);
  int y = get_texture_height(source);
  if (coord.x<0.0f)  coord.x = -coord.x;
  if (coord.x>=x)     coord.x = 2.0f*x - coord.x - 1;
  if (coord.y<0.0f)  coord.y = -coord.y;
  if (coord.y>=y)     coord.y = 2.0f*y - coord.y - 1;
  return read_imagef(source, nearest_sampler, gid);
}

static inline float4 __attribute__((overloadable)) read_image(__read_only image2d_t source, float2 coords) {
  return read_imagef(source, linear_normalized_sampler, coords);
}

static inline void __attribute__((overloadable)) write_image(__write_only image2d_t destination, float4 color, int2 gid) {
  write_imagef(destination, gid, color);
}

// 3D
static inline float4 __attribute__((overloadable)) read_image(__read_only image3d_t source, int3 gid) {
  return read_imagef(source, nearest_sampler, (int4){gid.x, gid.y, gid.z, 0});
}

static inline float4 __attribute__((overloadable)) read_image(__read_only image3d_t source, float3 coords) {
  return read_imagef(source, linear_normalized_sampler, (float4){coords.x,coords.y,coords.z,0});
}

static inline float4 __attribute__((overloadable)) read_image(__read_only image3d_t source, float4 coords) {
  return read_imagef(source, linear_normalized_sampler, coords);
}

static inline void __attribute__((overloadable)) write_image(__write_only image3d_t destination, float4 color, int3 gid) {
  write_imagef(destination, (int4){gid.x,gid.y,gid.z,0}, color);
}


static inline  float3 compress(float3 rgb, float2 compression) {
  return  compression.x*rgb + compression.y;
}


#endif //DEHANCER_GPULIB_OPENCL_COMMON_H
