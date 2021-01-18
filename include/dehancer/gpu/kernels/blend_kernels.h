//
// Created by denn on 18.01.2021.
//

#ifndef DEHANCER_GPULIB_BLEND_KERNELS_H
#define DEHANCER_GPULIB_BLEND_KERNELS_H

#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/blend.h"

DHCR_KERNEL void  kernel_blend(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        texture2d_read_t      overlay DHCR_BIND_TEXTURE(2),
        DHCR_CONST_ARG    float_ref_t  opacity DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG      int_ref_t     mode DHCR_BIND_BUFFER(4)
){
  Texel2d tex; get_kernel_texel2d(destination,tex);
  if (!get_texel_boundary(tex)) return;
  
  float2 coords = get_texel_coords(tex);
  
  float4  base          = read_image(source, coords);
  float4  overlay_color = read_image(overlay, coords);
  
  DCHR_BlendingMode r_mode = (DCHR_BlendingMode)mode;
  
  float4 result = blend(base,overlay_color,r_mode,opacity);

  write_image(destination, result, tex.gid);
}

#endif //DEHANCER_GPULIB_BLEND_KERNELS_H
