//
// Created by denn on 29.12.2020.
//

#pragma once

#include <cmath>
#include "dehancer/gpu/kernels/cuda/texture1d.h"
#include "dehancer/gpu/kernels/cuda/texture2d.h"
#include "dehancer/gpu/kernels/cuda/texture3d.h"
#include "dehancer/gpu/kernels/cuda/cutil_math.h"

static const float3 kIMP_Y_YUV_factor = {0.2125, 0.7154, 0.0721};

typedef struct {
    int2 gid;
    int2 size;
} Texel2d;

inline __device__ __host__ void get_kernel_texel2d(dehancer::nvcc::texture2d<float4> destination, Texel2d& tex) {

  tex.gid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tex.gid.y = blockIdx.y * blockDim.y + threadIdx.y;

  tex.size.x = destination.get_width();
  tex.size.y = destination.get_height();
}

inline __device__ __host__  bool get_texel_boundary(Texel2d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y) {
    return false;
  }
  return true;
}

inline __device__ __host__  float2 get_texel_coords(Texel2d tex) {
  return (float2){(float)tex.gid.x / (float)(tex.size.x - 1),
                  (float)tex.gid.y / (float)(tex.size.y - 1)};
}

template<class T>
__device__ const T& clamp(const T& v, const T& lo, const T& hi )
{
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

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
