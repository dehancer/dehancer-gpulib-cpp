//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_COMMON_H
#define DEHANCER_GPULIB_COMMON_H

static __constant float3 kIMP_Y_YUV_factor = {0.2125, 0.7154, 0.0721};

__constant sampler_t linear_normalized_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

typedef struct  {
    int2 gid;
    int2 size;
} Texel2d;

#define get_kernel_texel2d(destination, tex) { \
  tex.gid =  (int2){get_global_id(0), get_global_id(1)}; \
  tex.size = (int2){get_image_width(destination), get_image_height(destination)}; \
}

static inline  bool get_texel_boundary(Texel2d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y) {
    return false;
  }
  return true;
}

static inline  float2 get_texel_coords(Texel2d tex) {
  return (float2){(float)tex.gid.x / (float)(tex.size.x - 1),
                  (float)tex.gid.y / (float)(tex.size.y - 1)};
}


static inline float4 sampledColor(
        __read_only image2d_t inTexture,
        __write_only image2d_t outTexture,
        int2 gid
){
  int w = get_image_width (outTexture);
  int h = get_image_height (outTexture);

  float2 coords = (float2)((float)gid.x / (float)(w - 1),
                           (float)gid.y / (float)(h - 1));

  float4 color = read_imagef(inTexture, linear_normalized_sampler, coords);

  return color;
}

__kernel void kernel_dehancer_pass(
        __read_only image2d_t  source,
        __write_only image2d_t destination
){

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float4  color = sampledColor(source, destination, gid);

  write_imagef(destination, gid, color);
}

__kernel void kernel_grid(int levels, __write_only image2d_t destination )
{

  int w = get_image_width (destination);
  int h = get_image_height (destination);

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float2 coords = (float2)((float)gid.x / (float)(w - 1),
                           (float)gid.y / (float)(h - 1));

  int num = levels*2;
  int index_x = (int)(coords.x*(num));
  int index_y = (int)(coords.y*(num));

  int index = clamp((index_y+index_x)%2,(int)(0),(int)(num));

  float ret = (float)(index);

  float4 color = {ret*coords.x,ret*coords.y,ret,1.0} ;

  write_imagef(destination, gid, color);

}

#endif //DEHANCER_GPULIB_COMMON_H
