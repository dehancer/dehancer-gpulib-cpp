//
// Created by denn on 23.01.2021.
//

#ifndef DEHANCER_GPULIB_GAMMA_H
#define DEHANCER_GPULIB_GAMMA_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

static inline DHCR_DEVICE_FUNC float gamma_forward_channel(float x, DHCR_GammaParameters params) {
  //
  // https://en.wikipedia.org/wiki/Rec._709
  //
  if (x < params.lin_side_break) return x * params.lin_side_coeff;
  return  params.lin_side_slope * powf(x , params.base) - params.lin_side_offset;
}

static inline  DHCR_DEVICE_FUNC float gamma_inverse_channel(float x, DHCR_GammaParameters params) {
  //
  // https://en.wikipedia.org/wiki/Rec._709
  //
  if (x < params.gamma_side_break) return x / params.lin_side_coeff;
  return powf((x + params.lin_side_offset) / params.lin_side_slope, 1.0f / params.base);
}


#endif //DEHANCER_GPULIB_GAMMA_H
