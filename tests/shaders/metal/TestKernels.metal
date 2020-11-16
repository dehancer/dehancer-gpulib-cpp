//
// Created by denn nevera on 16/11/2020.
//

#include <metal_stdlib>
#include "aoBenchKernel.h"

using namespace metal;

kernel void ao_bench_kernel(
        texture2d<float, access::write>   destination [[texture(0)]],
        constant int&                   nsubsamples [[buffer (0)]],
        uint2 gid [[thread_position_in_grid]])
{
  int w = destination.get_width();
  int h = destination.get_height();

  int x = gid.x;
  int y = gid.y;

  float4 color = ao_bench(nsubsamples, x, y, w, h);

  destination.write(color, gid);
}
