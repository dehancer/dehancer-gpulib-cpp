//
// Created by denn on 09.01.2021.
//

#ifndef DEHANCER_GPULIB_UNARY_KERNELS_H
#define DEHANCER_GPULIB_UNARY_KERNELS_H

#include "dehancer/gpu/kernels/types.h"

DHCR_KERNEL void kernel_convolve_horizontal(
        DHCR_DEVICE_ARG     float*       scl DHCR_BIND_BUFFER(0),
        DHCR_DEVICE_ARG     float*       tcl DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG   int_ref_t         w DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG   int_ref_t         h DHCR_BIND_BUFFER(3),
        DHCR_DEVICE_ARG     float*   weights DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG    int_ref_t     size DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG    int_ref_t  address DHCR_BIND_BUFFER(6)
) {
  
  int2 tid; get_kernel_tid2d(tid);
  
  if ((tid.x < w) && (tid.y < h)) {
    
    float val = 0;
    const int index = ((tid.y * w) + tid.x);
    
    if (size==0) {
      tcl[index] = scl[index];
      return;
    }
    int half_size = size/2;

#pragma unroll
    for (int i = -half_size; i < half_size; ++i) {
      int jx =  tid.x+i;
/**
 * CLAMP address supports now
 */
      float f = 1;
      switch ((DHCR_EdgeAddress)address) {
        case DHCR_ADDRESS_CLAMP:
          if (jx<0)  jx = 0;
          if (jx>=w) jx = w-1;
          break;
        case DHCR_ADDRESS_BORDER:
          if (jx<0)  {jx = 0;f = 0;}
          if (jx>=w) {jx = w-1;f = 0;}
          break;
        case DHCR_ADDRESS_WRAP:
          if (jx<0)  jx -= i;
          if (jx>=w) jx -= i;
          break;
      }
      const int j = ((tid.y * w) + jx);
      val += scl[j] * weights[i+half_size] * f;
    }
    
    tcl[index] = val;
  }
}

DHCR_KERNEL void kernel_convolve_vertical (
        DHCR_DEVICE_ARG     float*       scl DHCR_BIND_BUFFER(0),
        DHCR_DEVICE_ARG     float*       tcl DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG    int_ref_t        w DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG    int_ref_t        h DHCR_BIND_BUFFER(3),
        DHCR_DEVICE_ARG      float*  weights DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG    int_ref_t     size DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG    int_ref_t  address DHCR_BIND_BUFFER(6)
) {
  
  int2 tid; get_kernel_tid2d(tid);
  
  if ((tid.x < w) && (tid.y < h)) {
    
    float val = 0;
    const int index = ((tid.y * w) + tid.x);
    
    if (size==0) {
      tcl[index] = scl[index];
      return;
    }
    int half_size = size/2;

#pragma unroll
    for (int i = -half_size; i < half_size; ++i) {
      int jy =  tid.y+i;
/**
 * CLAMP address supports now
 */
      float f = 1.0f;
      switch ((DHCR_EdgeAddress)address) {
        case DHCR_ADDRESS_CLAMP:
          if (jy<0)  jy = 0;
          if (jy>=h) jy = h-1;
          break;
        case DHCR_ADDRESS_BORDER:
          if (jy<0)  {jy = 0;f = 0;}
          if (jy>=h) {jy = h-1;f = 0;}
          break;
        case DHCR_ADDRESS_WRAP:
          if (jy<0)  jy -= i + half_size;
          if (jy>=h) jy -= i - half_size;
          break;
      }
      const int j = ((jy * w) + tid.x);
      val += scl[j] * weights[i+half_size] * f;
    }
    
    tcl[index] = val;
  }
}

#endif //DEHANCER_GPULIB_UNARY_KERNELS_H
