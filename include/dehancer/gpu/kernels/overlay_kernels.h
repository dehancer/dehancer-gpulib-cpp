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
        DHCR_CONST_ARG     bool_ref_t is_h_flipped DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG     bool_ref_t is_v_flipped DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG    float2_ref_t  offset DHCR_BIND_BUFFER(6)

        DHCR_KERNEL_GID_2D
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return;
  
  Texel2d tex_src; get_kernel_texel2d(source,tex_src);
  Texel2d tex_ovr; get_kernel_texel2d(overlay,tex_ovr);
  
  float4  base ;
  float4  overlay_color ;
  
  base          = sampled_color(source, tex.size, tex.gid);
  
  int2 coords = tex.gid-to_int2(offset);
  
  if (coords.x>=0 && coords.y>=0 && coords.x<tex_ovr.size.x && coords.y<tex_ovr.size.y) {
    coords = make_int2(
            is_h_flipped ? tex_ovr.size.x-coords.x : coords.x,
            is_v_flipped ? tex_ovr.size.y-coords.y : coords.y);
    overlay_color = sampled_color(overlay, tex_ovr.size, coords);
  }
  else
    overlay_color = to_float4(0.0f);
  
  float4 result = mix(base, overlay_color, overlay_color.w);

  float4 mask_rgba = to_float4(opacity);
  
  result = blend(base, result, DHCR_Normal, mask_rgba);
  
  write_image(destination, result, tex.gid);
}

#endif //DEHANCER_GPULIB_OVERLAY_KERNELS_H
