//
// Created by denn on 19.01.2021.
//

#ifndef DEHANCER_GPULIB_RESAMPLE_KERNELS_H
#define DEHANCER_GPULIB_RESAMPLE_KERNELS_H

#include "dehancer/gpu/kernels/resample.h"

DHCR_KERNEL void kernel_rotate90(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG int_ref_t    up DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_2D
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  int2 gid = make_int2(tex.gid.x, tex.gid.y);
  
  if (up == 1)      {gid.y = tex.gid.x; gid.x = tex.size.y-tex.gid.y-1; }
  else if (up == 2) {gid.y = tex.size.x-tex.gid.x-1; gid.x = tex.gid.y; }
  
  
  float4 rgb  =  read_image(source, gid);
  
  write_image(destination, rgb, tex.gid);
}

DHCR_KERNEL void kernel_flip(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG bool_ref_t    horizon DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG bool_ref_t   vertical DHCR_BIND_BUFFER(3)
        DHCR_KERNEL_GID_2D
){
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  int2 gid = make_int2(tex.gid.x, tex.gid.y);

  if (horizon)  gid.x = tex.size.x-gid.x-1;
  if (vertical) gid.y = tex.size.y-gid.y-1;
  
  float4 rgb  =  read_image(source, gid);
  
  write_image(destination, rgb, tex.gid);
}

DHCR_KERNEL void kernel_crop(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG int_ref_t   origin_left DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG int_ref_t    origin_top DHCR_BIND_BUFFER(3)
        DHCR_KERNEL_GID_2D
){
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  int2 gid = make_int2(tex.gid.x + origin_left, tex.gid.y + origin_top);
  
  float4 rgb  =  read_image(source, gid);
  
  write_image(destination, rgb, tex.gid);
}

DHCR_KERNEL void kernel_bilinear(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1)
        DHCR_KERNEL_GID_2D
){
  Texel2d tex_dest; get_kernel_texel2d(destination,tex_dest);
  
  if (!get_texel_boundary(tex_dest)) return;
  
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  float2 coords = get_texel_coords(tex_dest) * make_float2(tex_src.size.x-1,tex_src.size.y-1);
  
  float4 color = tex2D_bilinear(source, coords.x, coords.y);
  
  write_image(destination, color, tex_dest.gid);
}


DHCR_KERNEL void kernel_bicubic(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1)
        DHCR_KERNEL_GID_2D
){
  Texel2d tex_dest; get_kernel_texel2d(destination,tex_dest);
  
  if (!get_texel_boundary(tex_dest)) return;
  
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  float2 coords = get_texel_coords(tex_dest) * make_float2(tex_src.size.x-1,tex_src.size.y-1);
  
  float4 color = tex2D_bicubic(source, coords.x, coords.y);
  
  write_image(destination, color, tex_dest.gid);
}

DHCR_KERNEL void kernel_box_average(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1)
        DHCR_KERNEL_GID_2D
){
  Texel2d tex_dest; get_kernel_texel2d(destination,tex_dest);
  
  if (!get_texel_boundary(tex_dest)) return;
  
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  float2 coords = get_texel_coords(tex_dest) * make_float2(tex_src.size.x-1,tex_src.size.y-1);
  
  float4 color = tex2D_box_average(source, coords.x, coords.y);
  
  write_image(destination, color, tex_dest.gid);
}

#endif //DEHANCER_GPULIB_RESAMPLE_KERNELS_H
