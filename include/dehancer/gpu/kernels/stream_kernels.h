//
// Created by denn on 10.05.2021.
//

#ifndef DEHANCER_GPULIB_STREAM_KERNELS_H
#define DEHANCER_GPULIB_STREAM_KERNELS_H

#include "dehancer/gpu/kernels/stream_space.h"

DHCR_KERNEL void  kernel_stream_transform_ext(
        texture2d_read_t         source DHCR_BIND_TEXTURE(0),
        texture2d_write_t   destination DHCR_BIND_TEXTURE(1),
        texture3d_read_t  transform_lut DHCR_BIND_TEXTURE(2),
        DHCR_CONST_ARG_REF (DHCR_GammaParameters)             gamma_params DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG_REF (DHCR_LogParameters)                 log_params DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG  float4x4_ref_t                     cs_forward_matrix DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG  float4x4_ref_t                     cs_inverse_matrix DHCR_BIND_BUFFER(6),
        DHCR_CONST_ARG_REF (DHCR_TransformDirection)             direction DHCR_BIND_BUFFER(7),
        DHCR_CONST_ARG  bool_ref_t                   transform_lut_enabled DHCR_BIND_BUFFER(8),
        DHCR_CONST_ARG  bool_ref_t              transform_function_enabled DHCR_BIND_BUFFER(9),
        DHCR_CONST_ARG  float_ref_t                                 impact DHCR_BIND_BUFFER(10)
        DHCR_KERNEL_GID_2D
) {
  
  Texel2d tex;
  get_kernel_texel2d(destination, tex);
  if (!get_texel_boundary(tex)) return;
  
  float4 inColor = sampled_color(source, tex.size, tex.gid);
  
  float4 color  = inColor;
  
  if (transform_function_enabled) {
    color = transform_extended(color, gamma_params, log_params, cs_forward_matrix, cs_inverse_matrix, direction);
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


DHCR_KERNEL void kernel_bgr8_to_texture(
        DHCR_DEVICE_ARG uint8_t*        p_Input DHCR_BIND_BUFFER(0),
        DHCR_CONST_ARG  float_ref_t  scale DHCR_BIND_BUFFER(1),
        texture2d_write_t      destination DHCR_BIND_TEXTURE(2)
        
        DHCR_KERNEL_GID_2D
){
  
  Texel2d tex; get_kernel_texel2d(destination, tex);
  
  if (!get_texel_boundary(tex)) return;
  
  const int index = ((tex.gid.y * tex.size.x) + tex.gid.x) * 3;
  float4 color = make_float4(
          (float)p_Input[index + 2]*scale,
          (float)p_Input[index + 1]*scale,
          (float)p_Input[index + 0]*scale,
          1.0f);
  write_image(destination, color, tex.gid);
}

#endif //DEHANCER_GPULIB_STREAM_KERNELS_H
