//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_STD_KERNELS_H
#define DEHANCER_GPULIB_STD_KERNELS_H

#include "dehancer/gpu/kernels/resample.h"

/***
 * Bilinear sampler
 * @param source
 * @param destination_size
 * @param gid
 * @return
 */
static inline float4 __attribute__((overloadable)) sampled_color(
        __read_only image2d_t source,
        int2 destination_size,
        int2 gid
){
  
  int2 size = (int2){get_image_width(source), get_image_height(source)};
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y - 1)};
    coords = coords * (to_float2(size)-1.0f);
    return tex2D_bilinear(source, coords.x, coords.y);
  }
}


/***
 * Bicubic sampler
 * @param source
 * @param destination
 * @param gid
 * @return
 */
static inline float4 __attribute__((overloadable)) bicubic_sampled_color(
        __read_only image2d_t source,
        int2 destination_size,
        int2 gid
){
  int2 size = (int2){get_image_width(source), get_image_height(source)};
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y - 1)};
    coords = coords * (to_float2(size)-1.0f);
    return tex2D_bicubic(source, coords.x, coords.y);
  }
}

/***
 * Bicubic sampler
 * @param source
 * @param destination
 * @param gid
 * @return
 */
static inline float4 __attribute__((overloadable)) smooth_bicubic_sampled_color(
  __read_only image2d_t source,
  int2 destination_size,
  int2 gid
){
  int2 size = (int2){get_image_width(source), get_image_height(source)};

  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y - 1)};
    coords = coords * (to_float2(size)-1.0f);
    return tex2D_smooth_bicubic(source, coords.x, coords.y);
  }
}

/***
 * Box average sampler
 * @param source
 * @param destination
 * @param gid
 * @return
 */
static inline float4 __attribute__((overloadable)) box_average_sampled_color(
        __read_only image2d_t source,
        int2 destination_size,
        int2 gid
){
  int2 size = (int2){get_image_width(source), get_image_height(source)};
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y - 1)};
    coords = coords * (to_float2(size)-1.0f);
    return tex2D_box_average(source, coords.x, coords.y);
  }
}

/***
 * Pass kernel
 * @param source
 * @param destination
 * @return
 */
__kernel void kernel_dehancer_pass(
        __read_only image2d_t  source,
        __write_only image2d_t destination
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4  color = sampled_color(source, tex.size, tex.gid);

  write_imagef(destination, tex.gid, color);
}

#endif //DEHANCER_GPULIB_STD_KERNELS_H
