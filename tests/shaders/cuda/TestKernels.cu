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