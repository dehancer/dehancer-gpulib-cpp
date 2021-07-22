//
// Created by denn on 18.01.2021.
//

#ifndef DEHANCER_GPULIB_BLEND_KERNELS_H
#define DEHANCER_GPULIB_BLEND_KERNELS_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/blend.h"
#include "dehancer/gpu/kernels/resample.h"

DHCR_KERNEL void  kernel_blend(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        texture2d_read_t      overlay DHCR_BIND_TEXTURE(2),
        DHCR_CONST_ARG     bool_ref_t has_mask DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG    float_ref_t  opacity DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG      int_ref_t     mode DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG      int_ref_t int_mode DHCR_BIND_BUFFER(6),
        texture2d_read_t         mask DHCR_BIND_TEXTURE(7)
        DHCR_KERNEL_GID_2D
){
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return;
  
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  float4  base ;
  float4  overlay_color;
  
  switch ((DHCR_InterpolationMode)int_mode) {
    case DHCR_Bilinear:
      base          = sampled_color(source, tex.size, tex.gid);
      overlay_color = sampled_color(overlay, tex.size, tex.gid);
      break;
  
    case DHCR_Bicubic:
      base          = bicubic_sampled_color(source, tex.size, tex.gid);
      overlay_color = bicubic_sampled_color(overlay, tex.size, tex.gid);
      break;
  
    case DHCR_BoxAverage:
      base          = box_average_sampled_color(source, tex.size, tex.gid);
      overlay_color = box_average_sampled_color(overlay, tex.size, tex.gid);
      break;
  }
  
  float4 mask_rgba = to_float4(1.0f);
  
  if (has_mask) {
    mask_rgba = bicubic_sampled_color(mask, tex.size, tex.gid) * to_float4(opacity);
  }
  else
    mask_rgba = mask_rgba * to_float4(opacity);
  
  overlay_color = mix(base,overlay_color,overlay_color.w);
  
  float4 result = blend(base, overlay_color, (DHCR_BlendingMode)mode, mask_rgba);

  write_image(destination, result, tex.gid);
}


DHCR_KERNEL void kernel_grid(
        DHCR_CONST_ARG      int_ref_t levels DHCR_BIND_BUFFER(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1)
        DHCR_KERNEL_GID_2D
        )
{
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return;
  
  int w = tex.size.x;
  int h = tex.size.y;
  
  int2 gid = tex.gid;

  float2 coords = make_float2(
          (float)gid.x / (float)(w - 1),
          (float)gid.y / (float)(h - 1)
          );

  int num = 6*2;
  int index_x = (int)(coords.x*(num));
  int index_y = (int)(coords.y*(num));
  
  int index = clamp((index_y+index_x)%2,(int)(0),(int)(num));
  
  float ret = (float)(index);
  
  float4 color = make_float4(ret*coords.x, ret*coords.y, ret, 1.0);
  
  write_image(destination, color, tex.gid);
}

#endif //DEHANCER_GPULIB_BLEND_KERNELS_H
