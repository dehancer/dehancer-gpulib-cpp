//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_STD_KERNELS_H
#define DEHANCER_GPULIB_STD_KERNELS_H

#include "dehancer/gpu/kernels/opencl/common.h"
#include "dehancer/gpu/kernels/resample.h"

static inline float4 sampled_color(
        image2d_t source,
        image2d_t destination,
        int2 gid
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return make_float4(0.0f);
  
  if (tex.size.y==get_image_height(source) && get_image_width(source)==tex.size.x)
    return read_image(source, tex.gid);
  else {
    Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
    float2 coords = get_texel_coords(tex) * make_float2(tex_src.size);
    return tex2D_bilinear(source, coords.x, coords.y);
  }
}

static inline float4 bicubic_sampled_color(
        image2d_t source,
        image2d_t destination,
        int2 gid
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return make_float4(0.0f);
  
  if (tex.size.y==get_image_height(source) && get_image_width(source)==tex.size.x)
    return read_image(source, tex.gid);
  else {
    Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
    float2 coords = get_texel_coords(tex) * make_float2(tex_src.size);
    return tex2D_bicubic(source, coords.x, coords.y);
  }
}

static inline float4 box_average_sampled_color(
        image2d_t source,
        image2d_t destination,
        int2 gid
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return make_float4(0.0f);
  
  if (tex.size.y==get_image_height(source) && get_image_width(source)==tex.size.x)
    return read_image(source, tex.gid);
  else {
    Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
    float2 coords = get_texel_coords(tex) * make_float2(tex_src.size);
    return tex2D_box_average(source, coords.x, coords.y);
  }
}

__kernel void kernel_dehancer_pass(
        __read_only image2d_t  source,
        __write_only image2d_t destination
){
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float2 coords = get_texel_coords(tex);
  float4 color = read_image(source, coords);
  write_imagef(destination, tex.gid, color);
}

__kernel void kernel_grid(int levels, __write_only image2d_t destination )
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
