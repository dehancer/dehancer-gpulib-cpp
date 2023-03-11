//
// Created by denn on 19.01.2021.
//

#ifndef DEHANCER_GPULIB_RESAMPLE_KERNELS_H
#define DEHANCER_GPULIB_RESAMPLE_KERNELS_H

#include "dehancer/gpu/kernels/resample.h"

DHCR_KERNEL void kernel_crop(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG int_ref_t   origin_left DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG int_ref_t    origin_top DHCR_BIND_BUFFER(3)
        DHCR_KERNEL_GID_2D
){
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
//  if (tex.gid.x >= (tex.size.x - origin_left - origin_right)
//      ||
//      tex.gid.y >= (tex.size.y - origin_top - origin_bottom)
//      ||
//      tex.gid.x < origin_left
//      ||
//      tex.gid.y < origin_top
//          ) {
//    return ;
//  }
  
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
