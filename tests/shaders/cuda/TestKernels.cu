//
// Created by denn nevera on 16/11/2020.
//

#include <math.h>

template<class T>
__device__ const T& clamp(const T& v, const T& lo, const T& hi )
{
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

extern "C" __global__ void kernel_surface_gen(int levels, cudaSurfaceObject_t target, int w, int h)
{

  // Calculate surface coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= w || y >= h) {
    return ;
  }

  int2 gid = {x, y};

  float2 coords = (float2){(float)gid.x / (float)(w - 1),
                           (float)gid.y / (float)(h - 1)};

  int num = levels*2;
  int index_x = (int)(coords.x*(num));
  int index_y = (int)(coords.y*(num));

  int index = clamp((index_y+index_x)%2,0,num);

  float ret = (float)(index);

  float4 color = {ret*coords.x,ret*coords.y,ret,1.0} ;

  surf2Dwrite<float4>(color, target, x * sizeof(float4) , y , cudaBoundaryModeClamp);
}

extern "C" __global__ void kernel_vec_add(float* A, float* B, float* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}