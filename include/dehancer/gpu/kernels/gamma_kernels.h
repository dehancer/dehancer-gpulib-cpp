//
// Created by denn on 23.01.2021.
//

#ifndef DEHANCER_GPULIB_GAMMA_KERNELS_H
#define DEHANCER_GPULIB_GAMMA_KERNELS_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/gamma.h"

inline  DHCR_DEVICE_FUNC float3 apply_gama_forward(float3 in, DHCR_GammaParameters params) {
  float3 out;
  out.x = gamma_forward_channel(in.x, params);
  out.y = gamma_forward_channel(in.y, params);
  out.z = gamma_forward_channel(in.z, params);
  return out;
}

inline  DHCR_DEVICE_FUNC float3 apply_gama_inverse(float3 in, DHCR_GammaParameters params) {
  float3 out;
  out.x = gamma_inverse_channel(in.x, params);
  out.y = gamma_inverse_channel(in.y, params);
  out.z = gamma_inverse_channel(in.z, params);
  return out;
}

DHCR_KERNEL void  kernel_gamma(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        DHCR_CONST_ARG     DHCR_GammaParameters    params DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG  DHCR_TransformDirection direction DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG  float impact DHCR_BIND_BUFFER(4)
) {
  
  Texel2d tex; get_kernel_texel2d(destination, tex);
  if (!get_texel_boundary(tex)) return;
  
  float2 coords = get_texel_coords(tex);
  
  float4 rgba = sampled_color(source, destination, tex.gid);
  float3 result = make_float3(rgba);
  
  if (direction == DHCR_Forward) {
    result = apply_gama_forward(result, params);
  }
  else if (direction == DHCR_Inverse) {
    result = apply_gama_inverse(result, params);
  }

  result = mix(make_float3(rgba),result,make_float3(impact));
  
  write_image(destination, make_float4(result,rgba.w), tex.gid);
}

#endif //DEHANCER_GPULIB_GAMMA_KERNELS_H
