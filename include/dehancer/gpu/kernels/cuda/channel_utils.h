//
// Created by denn nevera on 30/11/2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/common.h"

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
        float* alphas)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = source.get_width();
  int h = source.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    
    float4 color     = read_image(source, gid);
    
    reds[index] = color.x;
    greens[index] = color.y;
    blues[index] = color.z;
    alphas[index] = color.w;
  }
}

extern "C" __global__ void channels_to_image (
        dehancer::nvcc::texture2d<float4> destination,
        float* reds,
        float* greens,
        float* blues,
        float* alphas)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = destination.get_width();
  int h = destination.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    float4 inColor = {reds[index], greens[index], blues[index], alphas[index]};
    write_image(destination, inColor, gid);
  }
}
