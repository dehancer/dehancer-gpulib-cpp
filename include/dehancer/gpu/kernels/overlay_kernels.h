//
// Created by denn on 18.01.2021.
//

#ifndef DEHANCER_GPULIB_OVERLAY_KERNELS_H
#define DEHANCER_GPULIB_OVERLAY_KERNELS_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/blend.h"
#include "dehancer/gpu/kernels/resample.h"

DHCR_KERNEL void  kernel_overlay_image(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        texture2d_read_t      overlay DHCR_BIND_TEXTURE(2),
        DHCR_CONST_ARG    float_ref_t  opacity DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG      int_ref_t int_mode DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG     bool_ref_t is_flipped DHCR_BIND_BUFFER(5)

        DHCR_KERNEL_GID_2D
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return;
  
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  float4  base ;
  //float4  overlay_color;
  
  uint p_Width = tex_src.size.x;
  uint p_Height = tex_src.size.y;
  
  float w = (float)get_texture_width(overlay);
  float h = (float)get_texture_height(overlay);
  
  float scale   = fmaxf(w/(float)(p_Width), h/(float)(p_Height));
  
  int2  id = tex.gid;
  float2 pos    = make_float2(id.x, is_flipped ? id.y : p_Height-id.y) * make_float2(1.0f/w, 1.0f/h) * scale;
  float2 size_i = make_float2(p_Width, p_Height) * make_float2(1.0f/w, 1.0f/h) * scale;
  float2 transl = make_float2(0.5f - size_i.x/2.0f, 0.5f - size_i.y/2.0f);
  
  switch ((DHCR_InterpolationMode)int_mode) {
    case DHCR_Bilinear:
      base          = sampled_color(source, tex.size, tex.gid);
      //overlay_color = sampled_color(overlay, tex.size, tex.gid);
      break;
  
    case DHCR_Bicubic:
      base          = bicubic_sampled_color(source, tex.size, tex.gid);
      //overlay_color = bicubic_sampled_color(overlay, tex.size, tex.gid);
      break;
  
    case DHCR_BoxAverage:
      base          = box_average_sampled_color(source, tex.size, tex.gid);
      //overlay_color = box_average_sampled_color(overlay, tex.size, tex.gid);
      break;
  }

  float4 overlay_color = read_image(overlay, pos+transl);
  
  float4 result = mix(base, overlay_color, overlay_color.w);

  float4 mask_rgba = make_float4(opacity);
  
  result = blend(base, result, DHCR_Normal, mask_rgba);

  write_image(destination, result, tex.gid);
}

#endif //DEHANCER_GPULIB_OVERLAY_KERNELS_H
