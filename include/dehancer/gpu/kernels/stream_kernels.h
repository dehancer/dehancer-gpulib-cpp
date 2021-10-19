//
// Created by denn on 10.05.2021.
//

#ifndef DEHANCER_GPULIB_STREAM_KERNELS_H
#define DEHANCER_GPULIB_STREAM_KERNELS_H

#include "dehancer/gpu/kernels/stream_space.h"

DHCR_KERNEL void  kernel_stream_transform(
        texture2d_read_t         source DHCR_BIND_TEXTURE(0),
        texture2d_write_t   destination DHCR_BIND_TEXTURE(1),
        texture3d_read_t  transform_lut DHCR_BIND_TEXTURE(2),
        DHCR_CONST_ARG_REF (DHCR_StreamSpace)                        space DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG_REF (DHCR_TransformDirection)             direction DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG  bool_ref_t                   transform_lut_enabled DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG  bool_ref_t              transform_function_enabled DHCR_BIND_BUFFER(6),
        DHCR_CONST_ARG  float_ref_t                                 impact DHCR_BIND_BUFFER(7)
        DHCR_KERNEL_GID_2D
) {
  
  Texel2d tex;
  get_kernel_texel2d(destination, tex);
  if (!get_texel_boundary(tex)) return;
  
  float4 inColor = sampled_color(source, tex.size, tex.gid);
  
  write_image(destination, inColor, tex.gid);
  
  float4 color  = inColor;

  if (transform_function_enabled) {
    color = transform(color, space, direction);
  }

  if (transform_lut_enabled) {
    // calibrated coeff
    if (direction == DHCR_Forward) {
      float4 a_low  = to_float4(-0.01f);
      float4 a_high = to_float4(1.009f);
      color = (color - a_low) / (a_high - a_low);
    }
    color = read_image(transform_lut, clamp(to_float3(color), 0.0f, 1.0f));
  }

  color = mix(clamp(inColor, 0.0f, 1.0f), color, impact);

  write_image(destination, to_float4(to_float3(color),inColor.w), tex.gid);
}

#endif //DEHANCER_GPULIB_STREAM_KERNELS_H
