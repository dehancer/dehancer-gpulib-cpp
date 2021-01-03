//
// Created by denn on 29.12.2020.
//

#pragma once

#include <cmath>
#include "dehancer/gpu/kernels/cuda/texture1d.h"
#include "dehancer/gpu/kernels/cuda/texture2d.h"
#include "dehancer/gpu/kernels/cuda/texture3d.h"
//#include "dehancer/gpu/kernels/cuda/cmath.h"
#include "dehancer/gpu/kernels/cuda/cmatrix.h"
#include "dehancer/gpu/kernels/constants.h"
#include "dehancer/gpu/kernels/types.h"

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

