//
// Created by denn on 03.01.2021.
//

#pragma once

#include "dehancer/gpu/kernels/resample.h"

/***
 * Bilinear sampler
 * @param source
 * @param destination_size
 * @param gid
 * @return
 */
inline __device__ __host__ float4 __attribute__((overloadable)) sampled_color(
        __read_only image2d_t source,
        int2 destination_size,
        int2 gid
){
  
  int2 size = make_int2(source.get_width(), source.get_height());
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = {(float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y - 1)};
    coords = coords * (make_float2(size)-1.0f);
    return tex2D_bilinear(source, coords.x, coords.y);
  }
}

/***
 * Bicubic sampler
 * @param source
 * @param destination_size
 * @param gid
 * @return
 */
inline __device__ __host__ float4 __attribute__((overloadable))  bicubic_sampled_color(
        __read_only image2d_t source,
        int2 destination_size,
        int2 gid
){
  
  int2 size = make_int2(source.get_width(), source.get_height());
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = make_float2((float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y- 1));
    coords = coords * (make_float2(size)-1.0f);
    return tex2D_bicubic(source, coords.x, coords.y);
  }
}

/***
 * Box average sampler
 * @param source
 * @param destination_size
 * @param gid
 * @return
 */
inline __device__ __host__ float4 __attribute__((overloadable)) box_average_sampled_color(
        __read_only image2d_t source,
        int2 destination_size,
        int2 gid
){
  
  int2 size = make_int2(source.get_width(), source.get_height());
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = make_float2((float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y- 1));
    coords = coords * (make_float2(size)-1.0f);
    return tex2D_box_average(source, coords.x, coords.y);
  }
}

/***
 * Pass kernel
 */
extern "C" __global__ void  kernel_dehancer_pass(
        __read_only image2d_t  source,
        __write_only image2d_t destination
){
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4  color = sampled_color(source, tex.size, tex.gid);
  
  write_image(destination, color, tex.gid);
}
