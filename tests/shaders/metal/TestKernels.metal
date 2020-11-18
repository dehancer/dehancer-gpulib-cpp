//
// Created by denn nevera on 16/11/2020.
//

#include <metal_stdlib>
#include "aoBenchKernel.h"

using namespace metal;

kernel void ao_bench_kernel(
        constant int&                     nsubsamples [[buffer (0)]],
        texture2d<float, access::write>   destination [[texture(1)]],
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

static constant float3 kIMP_Y_YUV_factor = {0.2125, 0.7154, 0.0721};
constexpr sampler baseSampler(address::clamp_to_edge, filter::linear, coord::normalized);
kernel void blend_kernel(
        texture2d<float, access::sample>   source [[texture(0)]],
        texture2d<float, access::write>   destination [[texture(1)]],
        device float*                     color_map   [[buffer (2)]],
        constant uint&                    levels      [[buffer (3)]],
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

  float4 inColor = source.sample(baseSampler, coords);

  float luminance = dot(inColor.rgb, kIMP_Y_YUV_factor);
  uint      index = clamp(uint(luminance*(float)(levels-1)),uint(0),uint(levels-1));
  float4    color = {1.0, 0.0, 0.0, 1.0};

  if (index<levels){
    color.r = color_map[index*3];
    color.g = color_map[index*3+1];
    color.b = color_map[index*3+2];
  }

  destination.write(color, gid);
}