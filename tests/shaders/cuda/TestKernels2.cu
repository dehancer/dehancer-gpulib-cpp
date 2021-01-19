//
// Created by denn nevera on 16/11/2020.
//

#include "dehancer/gpu/kernels/lib.h"

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