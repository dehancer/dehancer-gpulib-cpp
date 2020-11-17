#include "aoBenchKernel.h"

/* Compute the image for the scanlines from [y0,y1), for an overall image
   of width w and height h.
*/

__kernel void ao_bench_kernel(int nsubsamples, __write_only image2d_t destination )
{

  int w = get_image_width (destination);
  int h = get_image_height (destination);

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float4 color = ao_bench(nsubsamples, x, y, w, h);

  write_imagef(destination, gid, color);

}


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void blend_kernel(__read_only image2d_t source, __write_only image2d_t destination) {

  int2 gid = (int2)(get_global_id(0),
                    get_global_id(1));

  int2 imageSize = (int2)(get_image_width(destination),
                          get_image_height(destination));

  if (gid.x >= imageSize.x || gid.y >= imageSize.y)
  {
    return;
  }

  // Normalize coordinates
  float2 coords = (float2)((float)gid.x / (imageSize.x - 1),
                           (float)gid.y / (imageSize.y - 1));


  float4 inColor = read_imagef(source, sampler, coords);
  inColor.b = 0.5;

  write_imagef(destination, gid, inColor);

}