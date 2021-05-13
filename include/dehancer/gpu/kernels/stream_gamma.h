//
// Created by denn on 09.05.2021.
//

#ifndef DEHANCER_GPULIB_STREAM_GAMMA_H
#define DEHANCER_GPULIB_STREAM_GAMMA_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/gamma.h"


static inline DHCR_DEVICE_FUNC
float4 apply_gamma_forward( float4 in, DHCR_GammaParameters params) {
  float4 out;
#if DEHANCER_GPU_CODE == 1
  out.x = gamma_forward_channel(in.x, params);
  out.y = gamma_forward_channel(in.y, params);
  out.z = gamma_forward_channel(in.z, params);
  out.w = gamma_forward_channel(in.w, params);
#else
  out.x() = gamma_forward_channel(in.x(), params);
  out.y() = gamma_forward_channel(in.y(), params);
  out.z() = gamma_forward_channel(in.z(), params);
  out.w() = gamma_forward_channel(in.w(), params);
#endif
  return out;
}

static inline DHCR_DEVICE_FUNC
float4 apply_gamma_inverse( float4 in, DHCR_GammaParameters params) {
  float4 out;
#if DEHANCER_GPU_CODE == 1
  out.x = gamma_inverse_channel(in.x, params);
  out.y = gamma_inverse_channel(in.y, params);
  out.z = gamma_inverse_channel(in.z, params);
  out.w = gamma_inverse_channel(in.w, params);
#else
  out.x() = gamma_inverse_channel(in.x(), params);
  out.y() = gamma_inverse_channel(in.y(), params);
  out.z() = gamma_inverse_channel(in.z(), params);
  out.w() = gamma_inverse_channel(in.w(), params);
#endif
  
  return out;
}

#endif //DEHANCER_GPULIB_STREAM_GAMMA_H
