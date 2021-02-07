//
// Created by denn on 03.01.2021.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/common.h"
#include "dehancer/gpu/kernels/resample.h"

inline __device__ __host__ float4 sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){
  
  int2 size = (int2){source.get_width(), source.get_height()};
  
  if (size.y==destination.get_height() && destination.get_width()==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination.get_width() - 1),
                             (float)gid.y / (float)(destination.get_height()- 1)};
    coords = coords * make_float2(size);
    return tex2D_bilinear(source, coords.x, coords.y);
  }
}

inline __device__ __host__ float4 bicubic_sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){
  
  int2 size = (int2){source.get_width(), source.get_height()};
  
  if (size.y==destination.get_height() && destination.get_width()==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination.get_width() - 1),
                             (float)gid.y / (float)(destination.get_height()- 1)};
    coords = coords * make_float2(size);
    return tex2D_bicubic(source, coords.x, coords.y);
  }
}

inline __device__ __host__ float4 box_average_sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){
  
  int2 size = (int2){source.get_width(), source.get_height()};
  
  if (size.y==destination.get_height() && destination.get_width()==size.x)
    return read_image(source, gid);
  else {
    float2 coords = (float2){(float)gid.x / (float)(destination.get_width() - 1),
                             (float)gid.y / (float)(destination.get_height()- 1)};
    coords = coords * make_float2(size);
    return tex2D_box_average(source, coords.x, coords.y);
  }
}

extern "C" __global__ void  kernel_dehancer_pass(
        __read_only image2d_t  source,
        __write_only image2d_t destination
){
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4  color = sampled_color(source, destination, tex.gid);
  
  write_image(destination, color, tex.gid);
}

extern "C" __global__ void kernel_grid(int levels,
                                       __write_only image2d_t destination)
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
