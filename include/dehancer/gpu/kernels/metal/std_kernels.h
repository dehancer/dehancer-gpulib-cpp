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

static inline float4 __attribute__((overloadable)) sampled_color(
        texture2d_read_t source,
        int2 destination_size,
        int2 gid
){
  int2 size = int2(source.get_width(), source.get_height());

  if (size.y==destination_size.y && destination_size.x==size.x)
    return source.sample(nearest_sampler, (float2)(gid));
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
        texture2d_read_t source,
        int2 destination_size,
        int2 gid
){
  int2 size = int2(source.get_width(), source.get_height());
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y- 1)};
    coords = coords * (to_float2(size)-1.0f);
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
        texture2d_read_t source,
        int2 destination_size,
        int2 gid
){
  int2 size = int2(source.get_width(), source.get_height());
  
  if (size.y==destination_size.y && destination_size.x==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination_size.x - 1),
                             (float)gid.y / (float)(destination_size.y- 1)};
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
kernel void kernel_dehancer_pass(
        texture2d_read_t  source,
        texture2d_write_t destination,
        uint2 tid [[thread_position_in_grid]]
){
  
  int w = destination.get_width();
  int h = destination.get_height();
  
  int2 gid(tid.x,tid.y);
  
  if (gid.x>=w || gid.y>=h) return ;
  
  int2 size(w,h);
  
  float4  color = sampled_color(source, size, gid);
  
  destination.write(color,tid);
}