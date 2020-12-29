//
// Created by denn nevera on 16/11/2020.
//

#include "dehancer/gpu/kernels/cuda/common.h"


extern "C" __global__ void kernel_vec_add(float* A, float* B, float* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

extern "C" __global__ void kernel_grid_test_transform(
        dehancer::nvcc::texture2d<float4> source,
        dehancer::nvcc::texture2d<float4> destination
        )
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

  float4 color = source.read(coords * (float2){2.0f,2.0f});

  color.z = 0;

  destination.write(color, gid);

}