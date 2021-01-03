//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_COMMON_H
#define DEHANCER_GPULIB_COMMON_H

#include "dehancer/gpu/kernels/constants.h"
#include "dehancer/gpu/kernels/types.h"

__constant sampler_t linear_normalized_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define get_kernel_texel2d(destination, tex) { \
  tex.gid =  (int2){get_global_id(0), get_global_id(1)}; \
  tex.size = (int2){get_image_width(destination), get_image_height(destination)}; \
}

static inline  bool get_texel_boundary(Texel2d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y) {
    return false;
  }
  return true;
}

static inline  float2 get_texel_coords(Texel2d tex) {
  return (float2){(float)tex.gid.x / (float)(tex.size.x - 1),
                  (float)tex.gid.y / (float)(tex.size.y - 1)};
}

static inline float4 __attribute__((overloadable)) read_image(__read_only image2d_t source, int2 gid) {
  return read_imagef(source, nearest_sampler, gid);
}

static inline float4 __attribute__((overloadable)) read_image(__read_only image2d_t source, float2 coords) {
  return read_imagef(source, linear_normalized_sampler, coords);
}

static inline void __attribute__((overloadable)) write_image(__write_only image2d_t destination, float4 color, int2 gid) {
  write_imagef(destination, gid, color);
}

static inline void __attribute__((overloadable)) write_image(__write_only image3d_t destination, float4 color, int3 gid) {
  write_imagef(destination, (int4){gid.x,gid.y,gid.z,0}, color);
}

static inline void __attribute__((overloadable)) write_image(__write_only image1d_t destination, float4 color, int gid) {
  write_imagef(destination, gid, color);
}

static inline float4 sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){
  int w = get_image_width (destination);
  int h = get_image_height (destination);

  float2 coords = (float2)((float)gid.x / (float)(w - 1),
                           (float)gid.y / (float)(h - 1));

  float4 color = read_image(source, coords);

  return color;
}


#endif //DEHANCER_GPULIB_COMMON_H
