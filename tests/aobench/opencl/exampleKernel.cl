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


static __constant float3 kIMP_Y_YUV_factor = {0.2125, 0.7154, 0.0721};
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__kernel void blend_kernel(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        __global float* color_map,
        uint levels
) {

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

  float luminance = dot(inColor.rgb, kIMP_Y_YUV_factor);
  uint      index = clamp(uint(luminance*(float)(levels-1)),uint(0),uint(levels-1));
  float4    color = {1.0, 0.0, 0.0, 1.0};

  if (index<levels){
    color.r = color_map[index*3];
    color.g = color_map[index*3+1];
    color.b = color_map[index*3+2];
  }

  write_imagef(destination, gid, color);
}