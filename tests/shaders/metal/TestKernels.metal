//
// Created by denn nevera on 16/11/2020.
//

#include <metal_stdlib>
#include "aoBenchKernel.h"

using namespace metal;

kernel void ao_bench_kernel(
        texture2d<float, access::write>   destination [[texture(0)]],
        constant int&                   nsubsamples [[buffer (1)]],
        uint2 gid [[thread_position_in_grid]]
)
{
  int w = destination.get_width();
  int h = destination.get_height();

  int x = gid.x;
  int y = gid.y;

  float4 color = ao_bench(nsubsamples, x, y, w, h);

  destination.write(color, gid);
}

constexpr sampler baseSampler(address::clamp_to_edge, filter::linear, coord::normalized);

kernel void blend_kernel(
        texture2d<float, access::sample>   source [[texture(0)]],
        texture2d<float, access::write>   destination [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]
) {

  uint2 imageSize(destination.get_width(),destination.get_height());

  if (gid.x >= imageSize.x || gid.y >= imageSize.y)
  {
    return;
  }

  // Normalize coordinates
  float2 coords((float)gid.x / (imageSize.x - 1),
                (float)gid.y / (imageSize.y - 1));

  float4 color = source.sample(baseSampler, coords);

  color.b = 0.5;

  destination.write(color, gid);

}