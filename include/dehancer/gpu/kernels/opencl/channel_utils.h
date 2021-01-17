//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_CHANNEL_UTILS_H
#define DEHANCER_GPULIB_CHANNEL_UTILS_H

#include "dehancer/gpu/kernels/opencl/common.h"

__kernel void swap_channels_kernel (__global float* scl,
                                    __global float* tcl,
                                    int w,
                                    int h
                                    ) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    tcl[index] = scl[index];
  }
}

__kernel void image_to_channels (
        __read_only image2d_t source,
        __global float* reds,
        __global float* greens,
        __global float* blues,
        __global float* alphas,
        float4_ref_t slope,
        float4_ref_t offset,
        bool4_ref_t transform,
        int_ref_t direction
        )
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(source);
  int h = get_image_height(source);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);

    float4 color     = read_imagef(source, nearest_sampler, gid);
  
    if (transform.x)
      color.x = linearlog( color.x, slope.x, offset.x, direction);
  
    if (transform.y)
      color.y = linearlog( color.y, slope.y, offset.y, direction);
  
    if (transform.z)
      color.z = linearlog( color.z, slope.z, offset.z, direction);
  
    if (transform.w)
      color.w = linearlog( color.w, slope.w, offset.w, direction);
    
    reds[index] = color.x;
    greens[index] = color.y;
    blues[index] = color.z;
    alphas[index] = color.w;
  }

}

__kernel void channels_to_image (
        __write_only image2d_t destination,
        __global float* reds,
        __global float* greens,
        __global float* blues,
        __global float* alphas,
        float4_ref_t slope,
        float4_ref_t offset,
        bool4_ref_t transform,
        int_ref_t direction
        )
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(destination);
  int h = get_image_height(destination);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    float4 color = {reds[index], greens[index], blues[index], alphas[index]};
  
    if (transform.x)
      color.x = linearlog( color.x, slope.x, offset.x, direction);
  
    if (transform.y)
      color.y = linearlog( color.y, slope.y, offset.y, direction);
  
    if (transform.z)
      color.z = linearlog( color.z, slope.z, offset.z, direction);
  
    if (transform.w)
      color.w = linearlog( color.w, slope.w, offset.w, direction);
    
    write_imagef(destination, gid, color);
  }
}

#endif //DEHANCER_GPULIB_CHANNEL_UTILS_H
