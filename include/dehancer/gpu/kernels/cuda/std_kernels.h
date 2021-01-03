//
// Created by denn on 03.01.2021.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/common.h"

extern "C" __global__ void kernel_grid(int levels,
                                       dehancer::nvcc::texture2d<float4> destination)
{

  // Calculate surface coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int w = destination.get_width();
  int h = destination.get_height();

  if (x >= w || y >= h) {
    return ;
  }

  uint2 gid = (uint2){x, y};

  float2 coords = (float2){(float)gid.x / (float)(w - 1),
                           (float)gid.y / (float)(h - 1)};

  int num = levels*2;
  int index_x = (int)(coords.x*(float)(num));
  int index_y = (int)(coords.y*(float)(num));

  int index = clamp((index_y+index_x)%2,0,num);

  auto ret = (float)(index);

  float4 color = {ret*coords.x,ret*coords.y,ret,1.0} ;

  destination.write(color, gid);

}
