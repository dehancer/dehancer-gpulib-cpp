//
// Created by denn on 02.02.2021.
//

#include "dehancer/gpu/kernels/lib.h"
#include "dehancer/gpu/kernels/types.h"

#ifndef DEHANCER_VIDEO_MORPH_KERNELS_H
#define DEHANCER_VIDEO_MORPH_KERNELS_H

DHCR_KERNEL void kernel_dilate(
        texture2d_read_t               source DHCR_BIND_TEXTURE(0),
        texture2d_write_t         destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG int_ref_t         size DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG int2_ref_t        step DHCR_BIND_BUFFER(3)
) {
  
  Texel2d tex; get_kernel_texel2d(destination, tex);
  if (!get_texel_boundary(tex)) return;
  
  Texel2d tex_src; get_kernel_texel2d(source, tex_src);
  
  float2 coords = get_texel_coords(tex) * make_float2(tex_src.size);
  
  int2 gid = make_int2(coords);
  
  float4  color = make_float4(0.0f);
  
  #pragma unroll
  for (int j = -size; j <= size; ++j) {
    float4 c =  read_image(source, gid + make_int2(j, j) * step);
    color = fmaxf(color,c);
  }
  
  write_image(destination, color, tex.gid);
}

DHCR_KERNEL void kernel_erode(
        texture2d_read_t               source DHCR_BIND_TEXTURE(0),
        texture2d_write_t         destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG int_ref_t         size DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG int2_ref_t        step DHCR_BIND_BUFFER(3)
) {
  
  Texel2d tex; get_kernel_texel2d(destination, tex);
  if (!get_texel_boundary(tex)) return;
  
  Texel2d tex_src; get_kernel_texel2d(source, tex_src);
  
  float2 coords = get_texel_coords(tex) * make_float2(tex_src.size);
  
  int2 gid = make_int2(coords);
  
  float4  color = make_float4(1.0f);

#pragma unroll
  for (int j = -size; j <= size; ++j) {
    float4 c =  read_image(source, gid + make_int2(j, j) * step);
    color = fminf(color,c);
  }
  
  write_image(destination, color, tex.gid);
}

#endif //DEHANCER_VIDEO_MORPH_KERNELS_H
