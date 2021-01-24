//
// Created by denn nevera on 30/11/2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/common.h"
#include "dehancer/gpu/kernels/cuda/std_kernels.h"

extern "C" __global__ void swap_channels_kernel (
        float* scl,
        float* tcl,
        int w,
        int h)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    tcl[index] = scl[index];
  }
}

extern "C" __global__ void image_to_channels (
        dehancer::nvcc::texture2d<float4> source,
        float* reds,
        float* greens,
        float* blues,
        float* alphas
        ,
        float4_ref_t slope
        ,
        float4_ref_t offset
        ,
        bool4_ref_t transform
        ,
        TransformDirection direction
        ,
        bool_ref_t has_mask
        ,
        texture2d_read_t mask
        )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = source.get_width();
  int h = source.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    
    float4 color     = read_image(source, gid);
  
    float4  eColor = has_mask ? sampled_color(mask, source, gid) : make_float4(1.0f);
  
    if (transform.x)
      color.x = linearlog( color.x, slope.x, offset.x, direction, eColor.x);
  
    if (transform.y)
      color.y = linearlog( color.y, slope.y, offset.y, direction, eColor.y);
  
    if (transform.z)
      color.z = linearlog( color.z, slope.z, offset.z, direction, eColor.z);
  
    if (transform.w)
      color.w = linearlog( color.w, slope.w, offset.w, direction, eColor.w);
    
    reds[index]   = color.x;
    greens[index] = color.y;
    blues[index]  = color.z;
    alphas[index] = color.w;
  }
}

extern "C" __global__ void channels_to_image (
        dehancer::nvcc::texture2d<float4> destination,
        float* reds,
        float* greens,
        float* blues,
        float* alphas
        ,
        float4_ref_t slope
        ,
        float4_ref_t offset
        ,
        bool4_ref_t transform
        ,
        TransformDirection direction
        ,
        bool_ref_t has_mask
        ,
        texture2d_read_t mask
        )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = destination.get_width();
  int h = destination.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    
    const int index = ((gid.y * w) + gid.x);
    
    float4 color = make_float4(reds[index], greens[index], blues[index], alphas[index]);
  
    float4  eColor = has_mask ? sampled_color(mask, destination, gid) : make_float4(1.0f);
  
    if (transform.x)
      color.x = linearlog( color.x, slope.x, offset.x, direction, eColor.x);

    if (transform.y)
      color.y = linearlog( color.y, slope.y, offset.y, direction, eColor.y);

    if (transform.z)
      color.z = linearlog( color.z, slope.z, offset.z, direction, eColor.z);

    if (transform.w)
      color.w = linearlog( color.w, slope.w, offset.w, direction, eColor.w);
    
    write_image(destination, color, gid);
  }
}
