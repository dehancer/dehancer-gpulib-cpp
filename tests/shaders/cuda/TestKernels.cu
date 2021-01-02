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

extern "C" __global__ void kernel_vec_dev(float* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] /= 3.0f;
}

extern "C" __global__ void kernel_test_simple_transform(
        dehancer::nvcc::texture2d<float4> source,
        dehancer::nvcc::texture2d<float4> destination
) {

  // Calculate surface coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int w = destination.get_width();
  int h = destination.get_height();

  if (x >= w || y >= h) {
    return;
  }

  uint2 gid = (uint2) {x, y};

  float2 coords = (float2){(float)gid.x / (float)(w - 1),
                           (float)gid.y / (float)(h - 1)};

  float4 color = source.read(coords);

  color.x = 0;

  destination.write(color, gid);
}

extern "C" __global__ void kernel_grid_test_transform(
        dehancer::nvcc::texture2d<float4> source,
        dehancer::nvcc::texture2d<float4> destination,
        dehancer::nvcc::texture3d<float4> d3DLut,
        dehancer::nvcc::texture1d<float4> d1DLut
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

  color = d3DLut.read((float3){color.x,color.y,color.z});

  color.x = d1DLut.read(color.x).x;
  color.y = d1DLut.read(color.y).y;
  color.z = d1DLut.read(color.z).z;

  destination.write(color, gid);

}

inline __device__ float3 compress(float3 rgb, float2 compression) {
  return  compression.x*rgb + compression.y;
}

///
/// @brief Kernel optimized 3D LUT identity
///
extern "C" __global__  void kernel_make3DLut_transform(
        dehancer::nvcc::texture3d<float4> d3DLut,
        float2  compression)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;

  uint w = d3DLut.get_width();
  uint h = d3DLut.get_height();
  uint d = d3DLut.get_depth();

  if (x >= w || y >= h || z >= d) {
    return ;
  }

  uint3 gid = {x,y,z};

  float3 denom = (float3){d3DLut.get_width()-1, d3DLut.get_height()-1, d3DLut.get_depth()-1};
  float3 c = compress((float3){gid.x, gid.y, gid.z}/denom, compression);

  // transformation
  float4 color = (float4){c.x/2.f, c.y, 0.f, 1.f};

  d3DLut.write(color, gid);
}

///
/// @brief Kernel optimized 1D LUT identity
///
extern "C" __global__  void kernel_make1DLut_transform(
        dehancer::nvcc::texture1d<float4> d1DLut,
        float2  compression)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;

  uint w = d1DLut.get_width();

  if (x >= w) {
    return ;
  }

  float3 denom = (float3){w-1, w-1, w-1};
  float3 c = compress((float3){x, x, x}/denom, compression);

  // linear transform with compression
  float4 color = (float4){c.x, c.y, c.z, 1.f};

  d1DLut.write(color, x);
}