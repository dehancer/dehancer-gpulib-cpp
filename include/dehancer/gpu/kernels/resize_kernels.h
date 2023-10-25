//
// Created by denn on 19.01.2021.
//

#ifndef DEHANCER_GPULIB_RESIZE_KERNELS_H
#define DEHANCER_GPULIB_RESIZE_KERNELS_H

#include "dehancer/gpu/kernels/resample.h"

DHCR_KERNEL void kernel_gauss(
        texture2d_read_t            source DHCR_BIND_TEXTURE(0),
        texture2d_write_t           destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG float2_ref_t direction DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_2D
){
  Texel2d tex_dest; get_kernel_texel2d(destination,tex_dest);
  
  if (!get_texel_boundary(tex_dest)) return;
  
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  float2 uv = get_texel_coords(tex_dest);
  float2 res = make_float2(tex_src.size.x,tex_src.size.y);

  float4 color = to_float4(0.0f);
  float2 off1 = to_float2(1.3333333333333333f) * direction / res;
  color += read_image(source, uv) * 0.29411764705882354f;
  color += read_image(source, uv + (off1)) * 0.35294117647058826f;
  color += read_image(source, uv - (off1)) * 0.35294117647058826f;

  write_image(destination, color, tex_dest.gid);
}

DHCR_KERNEL void kernel_lanczos(
  texture2d_read_t            source DHCR_BIND_TEXTURE(0),
  texture2d_write_t           destination DHCR_BIND_TEXTURE(1),
  DHCR_CONST_ARG float2_ref_t direction DHCR_BIND_BUFFER(2)
  DHCR_KERNEL_GID_2D
){
  Texel2d tex_dest; get_kernel_texel2d(destination,tex_dest);

  if (!get_texel_boundary(tex_dest)) return;

  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);

  float2 uv = get_texel_coords(tex_dest);
  float2 res = make_float2(tex_src.size.x,tex_src.size.y);

  float4 color = to_float4(0.0f);
  float2 off1 = to_float2(1.0f) * direction / res;
  float2 off2 = to_float2(2.0f) * direction / res;
  float2 off3 = to_float2(3.0f) * direction / res;
  float2 off4 = to_float2(4.0f) * direction / res;

  color += read_image(source, uv) * 0.38026f;

  color += read_image(source, uv + (off1)) * 0.27667f;
  color += read_image(source, uv - (off1)) * 0.27667f;

  color += read_image(source, uv + (off2)) * 0.08074f;
  color += read_image(source, uv - (off2)) * 0.08074f;

  color += read_image(source, uv + (off3)) * -0.02612f;
  color += read_image(source, uv - (off3)) * -0.02612f;

  color += read_image(source, uv + (off4)) * -0.02143f;
  color += read_image(source, uv - (off4)) * -0.0214f;

  write_image(destination, color, tex_dest.gid);
}

#endif //DEHANCER_GPULIB_RESIZE_KERNELS_H
