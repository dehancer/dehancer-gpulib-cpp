//
// Created by denn on 24.04.2021.
//

#ifndef DEHANCER_GPULIB_CLUT_KERNELS_H
#define DEHANCER_GPULIB_CLUT_KERNELS_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

DHCR_KERNEL void kernel_make1DLut(
        texture1d_write_t         d1DLut DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t  compression DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_1D
) {
  Texel1d tex;
  get_kernel_texel1d(d1DLut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 denom = make_float3(get_texture_width(d1DLut)-1);
  float4 input_color  = make_float4(compress(make_float3(tex.gid)/denom, compression),1);
  
  write_image(d1DLut, input_color, tex.gid);
  
}

DHCR_KERNEL void kernel_make2DLut(
        texture2d_write_t         d2DLut DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t  compression DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG uint_ref_t  clevel DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex;
  get_kernel_texel2d(d2DLut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float qsize = (float)(clevel*clevel);
  float denom = qsize-1;
  
  uint  bindex =
          (floorf((float)(tex.gid.x) / denom)
           + (float)(clevel) * floorf( (float)(tex.gid.y)/denom));
  float b = (float)bindex/denom;
  
  float xindex = floor((float)(tex.gid.x) / qsize);
  float yindex = floor((float)(tex.gid.y) / qsize);
  float r = ((float)(tex.gid.x)-xindex*(qsize))/denom;
  float g = ((float)(tex.gid.y)-yindex*(qsize))/denom;
  
  float3 rgb = compress(make_float3(r,g,b),compression);
  
  write_image(d2DLut, make_float4(rgb,1), tex.gid);
  
}

DHCR_KERNEL void kernel_make3DLut(
        texture3d_write_t         d3DLut DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t  compression DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_3D
) {
  Texel3d tex;
  get_kernel_texel3d(d3DLut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 denom = make_float3(get_texture_width(d3DLut)-1,
                             get_texture_height(d3DLut)-1,
                             get_texture_depth(d3DLut)-1);
  
  float4 input_color  = make_float4(compress(make_float3(tex.gid)/denom, compression),1);
  
  write_image(d3DLut, input_color, tex.gid);
  
}

#endif //DEHANCER_GPULIB_CLUT_KERNELS_H
