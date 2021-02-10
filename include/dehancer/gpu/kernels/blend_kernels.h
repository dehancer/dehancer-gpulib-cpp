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
  
  float2 coords = get_texel_coords(tex) * make_float2(tex_src.size);
  
  float4  base ;
  float4  overlay_color;
  
  switch ((DHCR_InterpolationMode)int_mode) {
    case DHCR_Bilinear:
      base          = sampled_color(source, destination, tex.gid);
      overlay_color = sampled_color(overlay, destination, tex.gid);
      break;
  
    case DHCR_Bicubic:
      base          = bicubic_sampled_color(source, destination, tex.gid);
      overlay_color = bicubic_sampled_color(overlay, destination, tex.gid);
      break;
  
    case DHCR_BoxAverage:
      base          = box_average_sampled_color(source, destination, tex.gid);
      overlay_color = box_average_sampled_color(overlay, destination, tex.gid);
      break;
  }
  
  float4 mask_rgba = make_float4(1.0f);
  
  if (has_mask) {
    mask_rgba = bicubic_sampled_color(mask, destination, tex.gid) * make_float4(opacity);
  }
  else
    mask_rgba = mask_rgba * make_float4(opacity);
  
  float4 result = blend(base, overlay_color, (DHCR_BlendingMode)mode, mask_rgba);

  write_image(destination, result, tex.gid);
}

#endif //DEHANCER_GPULIB_BLEND_KERNELS_H
