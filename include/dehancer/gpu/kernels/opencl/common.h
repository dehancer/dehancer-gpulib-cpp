//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_COMMON_H
#define DEHANCER_GPULIB_COMMON_H

#include "dehancer/gpu/kernels/constants.h"
#include "dehancer/gpu/kernels/types.h"

__constant sampler_t linear_normalized_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define get_kernel_texel1d(destination, tex) { \
  tex.gid =  (int)get_global_id(0); \
  tex.size = (int)get_image_width(destination); \
}

#define get_kernel_texel2d(destination, tex) { \
  tex.gid =  (int2){get_global_id(0), get_global_id(1)}; \
  tex.size = (int2){get_image_width(destination), get_image_height(destination)}; \
}

#define get_kernel_texel3d(destination, tex) { \
  tex.gid =  (int3){get_global_id(0), get_global_id(1), get_global_id(2)}; \
  tex.size = (int3){get_image_width(destination), get_image_height(destination), get_image_depth(destination)}; \
}

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


static inline float4 sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){

  Texel2d tex; get_kernel_texel2d(destination,tex);

  float2 coords = get_texel_coords(tex);

  return read_image(source, coords);
}


#endif //DEHANCER_GPULIB_COMMON_H
