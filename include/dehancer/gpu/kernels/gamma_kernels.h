//
// Created by denn on 23.01.2021.
//

#ifndef DEHANCER_GPULIB_GAMMA_KERNELS_H
#define DEHANCER_GPULIB_GAMMA_KERNELS_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/gamma.h"

static inline  DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) apply_gamma_forward(float3 in, DHCR_GammaParameters params) {
  float3 out;
  out.x = gamma_forward_channel(in.x, params);
  out.y = gamma_forward_channel(in.y, params);
  out.z = gamma_forward_channel(in.z, params);
  return out;
}

static inline  DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) apply_gamma_inverse(float3 in, DHCR_GammaParameters params) {
  float3 out;
  out.x = gamma_inverse_channel(in.x, params);
  out.y = gamma_inverse_channel(in.y, params);
  out.z = gamma_inverse_channel(in.z, params);
  return out;
}

DHCR_KERNEL void  kernel_gamma(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG_REF (DHCR_GammaParameters)    params DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG_REF (DHCR_TransformDirection) direction DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG  float_ref_t impact DHCR_BIND_BUFFER(4)
        DHCR_KERNEL_GID_2D
) {

  Texel2d tex; get_kernel_texel2d(destination, tex);
  if (!get_texel_boundary(tex)) return;
  
  float4 inColor = sampled_color(source, tex.size, tex.gid);
  
  float3 rgb = to_float3(inColor);
  float3 result = rgb;

  if (direction == DHCR_Forward) {
    result = apply_gamma_forward(result, params);
  }
  else if (direction == DHCR_Inverse) {
    result = apply_gamma_inverse(result, params);
  }
 
  result = mix(rgb, result, to_float3(impact));
  inColor = to_float4(result, inColor.w);

  write_image(destination, inColor, tex.gid);
}

#endif //DEHANCER_GPULIB_GAMMA_KERNELS_H
