#include "dehancer/gpu/kernels/opencl/channel_utils.h"
#include "dehancer/gpu/kernels/opencl/blur_kernels.h"

__kernel void grid_kernel(int levels, __write_only image2d_t destination )
{

  int w = get_image_width (destination);
  int h = get_image_height (destination);

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float2 coords = (float2)((float)gid.x / (w - 1),
                           (float)gid.y / (h - 1));

  int num = levels*2;
  int index_x = int(coords.x*(num));
  int index_y = int(coords.y*(num));

  int index = clamp((index_y+index_x)%2,int(0),int(num));

  float ret = (float)(index);

  float4 color = {ret*coords.x,ret*coords.y,ret,1.0} ;

  write_imagef(destination, gid, color);

}