//
// Created by denn on 09.05.2021.
//

#ifndef DEHANCER_GPULIB_STREAM_GAMMA_H
#define DEHANCER_GPULIB_STREAM_GAMMA_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/gamma.h"

static inline DHCR_DEVICE_FUNC
float4 apply_gamma_forward(float4 in, DHCR_GammaParameters params) {
  float4 out;
  out[0] = gamma_forward_channel(in[0], params);
  out[1] = gamma_forward_channel(in[1], params);
  out[2] = gamma_forward_channel(in[2], params);
  out[3] = gamma_forward_channel(in[3], params);
  return out;
}

static inline DHCR_DEVICE_FUNC
float4 apply_gamma_inverse(float4 in, DHCR_GammaParameters params) {
  float4 out;
  out[0] = gamma_inverse_channel(in[0], params);
  out[1] = gamma_inverse_channel(in[1], params);
  out[2] = gamma_inverse_channel(in[2], params);
  out[3] = gamma_inverse_channel(in[3], params);
  return out;
}

#endif //DEHANCER_GPULIB_STREAM_GAMMA_H
