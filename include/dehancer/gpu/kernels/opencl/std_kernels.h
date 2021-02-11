//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_STD_KERNELS_H
#define DEHANCER_GPULIB_STD_KERNELS_H

#include "dehancer/gpu/kernels/opencl/common.h"
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
                             (float)gid.y / (float)(destination_size.y- 1)};
    coords = coords * make_float2(size);
    return tex2D_bilinear(source, coords.x, coords.y);
  }
}

static inline float4 __attribute__((overloadable)) sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){
  
  int2 size = (int2){get_image_width(source), get_image_height(source)};
  
  if (size.y==get_image_height(destination) && get_image_width(destination)==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(get_image_width(destination) - 1),
                             (float)gid.y / (float)(get_image_height(destination)- 1)};
    coords = coords * make_float2(size);
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
                             (float)gid.y / (float)(destination_size.y- 1)};
    coords = coords * make_float2(size);
    return tex2D_bicubic(source, coords.x, coords.y);
  }
}

static inline float4 __attribute__((overloadable)) bicubic_sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){
  int2 size = (int2){get_image_width(source), get_image_height(source)};
  
  if (size.y==get_image_height(destination) && get_image_width(destination)==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(get_image_width(destination) - 1),
                             (float)gid.y / (float)(get_image_height(destination)- 1)};
    coords = coords * make_float2(size);
    return tex2D_bicubic(source, coords.x, coords.y);
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
                             (float)gid.y / (float)(destination_size.y- 1)};
    coords = coords * make_float2(size);
    return tex2D_box_average(source, coords.x, coords.y);
  }
}

static inline float4 __attribute__((overloadable)) box_average_sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){
  int2 size = (int2){get_image_width(source), get_image_height(source)};
  
  if (size.y==get_image_height(destination) && get_image_width(destination)==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(get_image_width(destination) - 1),
                             (float)gid.y / (float)(get_image_height(destination)- 1)};
    coords = coords * make_float2(size);
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

__kernel void kernel_grid(
        int levels,
        __write_only image2d_t destination )
{
  
  int w = get_image_width (destination);
  int h = get_image_height (destination);
  
  int x = get_global_id(0);
  int y = get_global_id(1);
  
  int2 gid = (int2)(x, y);
  
  float2 coords = (float2)((float)gid.x / (float)(w - 1),
                           (float)gid.y / (float)(h - 1));
  
  int num = levels*2;
  int index_x = (int)(coords.x*(num));
  int index_y = (int)(coords.y*(num));
  
  int index = clamp((index_y+index_x)%2,(int)(0),(int)(num));
  
  float ret = (float)(index);
  
  float4 color = {ret*coords.x,ret*coords.y,ret,1.0} ;
  
  write_imagef(destination, gid, color);
  
}

#endif //DEHANCER_GPULIB_STD_KERNELS_H
