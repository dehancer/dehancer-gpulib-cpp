//
// Created by denn nevera on 16/11/2020.
//

#include "dehancer/gpu/kernels/cuda/cuda.h"

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
        __read_only image2d_t source,
        __write_only image2d_t destination
) {
  // Calculate surface coordinates
  Texel2d tex; get_kernel_texel2d(destination,tex);

  if (!get_texel_boundary(tex)) return;

  float2 coords = get_texel_coords(tex);

  float4 color = sampled_color(source, destination, tex.gid);

  color.x = 0;

  write_image(destination, color, tex.gid);
}

extern "C" __global__ void kernel_grid_test_transform(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        __read_only image3d_t d3DLut,
        __read_only image1d_t d1DLut
)
{
  // Calculate surface coordinates
  Texel2d tex; get_kernel_texel2d(destination,tex);

  if (!get_texel_boundary(tex)) return;

  float2 coords = get_texel_coords(tex);

  float4 color = sampled_color(source, destination, tex.gid);

  color = read_image(d3DLut, color);

  color = read_image(d1DLut, color);

  write_image(destination, color, tex.gid);
}

inline __device__ float3 compress(float3 rgb, float2 compression) {
  return  compression.x*rgb + compression.y;
}

///
/// @brief Kernel optimized 3D LUT identity
///
extern "C" __global__ void kernel_make3DLut_transform(
        __write_only image3d_t d3DLut,
        float2  compression
)
{

  Texel3d tex; get_kernel_texel3d(d3DLut,tex);

  if (!get_texel_boundary(tex)) return;

  float3 c = compress(get_texel_coords(tex), compression);

  // transformation
  float4 color = (float4){c.x/2.f, c.y, 0.f, 1.f};

  write_image(d3DLut, color, tex.gid);
}

///
/// @brief Kernel optimized 1D LUT identity
///
extern "C" __global__ void kernel_make1DLut_transform(
        __write_only image1d_t d1DLut [[ texture(0) ]],
        float2  compression)
{

  Texel1d tex; get_kernel_texel1d(d1DLut,tex);

  if (!get_texel_boundary(tex)) return;

  float3 denom = (float3){tex.size, tex.size, tex.size};

  float x = tex.gid;

  float3 c = compress((float3){x, x, x}/denom, compression);

  // linear transform with compression
  float4 color = (float4){c.x, c.y, c.z, 1.f};

  write_image(d1DLut, color, x);
}