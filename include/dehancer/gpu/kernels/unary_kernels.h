//
// Created by denn on 09.01.2021.
//

#ifndef DEHANCER_GPULIB_UNARY_KERNELS_H
#define DEHANCER_GPULIB_UNARY_KERNELS_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

DHCR_KERNEL void kernel_convolve_horizontal(
        DHCR_DEVICE_ARG     float*       scl DHCR_BIND_BUFFER(0),
        DHCR_DEVICE_ARG     float*       tcl DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG   int_ref_t         w DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG   int_ref_t         h DHCR_BIND_BUFFER(3),
        DHCR_DEVICE_ARG     float*   weights DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG    int_ref_t     size DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG    int_ref_t  address DHCR_BIND_BUFFER(6),
        DHCR_CONST_ARG int_ref_t channel_index DHCR_BIND_BUFFER(7)
        DHCR_KERNEL_GID_2D
) {
  
  int2 tid; get_kernel_tid2d(tid);
  
  if ((tid.x < w) && (tid.y < h)) {
    
    float val = 0;
    const int index = ( tid.y * w + tid.x );
    
    if (size==0) {
      tcl[index] = scl[index];
      return;
    }
    
    int half_size = size/2;

#pragma unroll 2
    for (int i = 0; i < half_size; ++i) {
      int jx  = tid.x + i;
      int jx2 = jx - half_size;
/**
 * CLAMP address supports now
 */
      float f = 1;
      switch ((DHCR_EdgeMode)address) {
        case DHCR_ADDRESS_CLAMP:
          if (jx2<0) jx2 = 0;
          if (jx>=w) jx  = w-1;
          break;
        case DHCR_ADDRESS_BORDER:
          if (jx2<0) { jx2 = 0;   f = 0;}
          if (jx>=w) { jx  = w-1; f = 0;}
          break;
        case DHCR_ADDRESS_WRAP:
          if (jx2<0)  jx2 = -jx2;
          if (jx2>=w) jx2 =  w - 1 - (jx2 - w);
    
          if (jx>=w)  jx  =  w - 1 - (jx - w);
          if (jx<0)   jx  = -jx;
          break;
      }
      
      int j = ((tid.y * w) + jx);
      int j2 = ((tid.y * w) + jx2);
      
      if (j>w*h) continue;
      if (j2>w*h) continue;
      
      val += scl[j] * weights[i+half_size] * f + scl[j2] * weights[i] * f;
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
        DHCR_CONST_ARG    int_ref_t  address DHCR_BIND_BUFFER(6),
        DHCR_CONST_ARG int_ref_t channel_index DHCR_BIND_BUFFER(7)
        DHCR_KERNEL_GID_2D
) {
  
  int2 tid; get_kernel_tid2d(tid);
  
  if ((tid.x < w) && (tid.y < h)) {
    
    float val = 0;
    const int index = ( tid.y * w + tid.x );
    
    if (size==0) {
      tcl[index] = scl[index];
      return;
    }
    int half_size = size/2;

#pragma unroll 2
    for (int i = 0; i < half_size; ++i) {
      int jy =  tid.y+i;
      int jy2 = jy - half_size;
/**
 * CLAMP address supports now
 */
      float f = 1.0f;
      switch ((DHCR_EdgeMode)address) {
        case DHCR_ADDRESS_CLAMP:
          if (jy2<0) jy2 = 0;
          if (jy>=h) jy  = h-1;
          break;
        case DHCR_ADDRESS_BORDER:
          if (jy2<0)  {jy2 = 0;   f = 0;}
          if (jy>=h)  {jy  = h-1; f = 0;}
          break;
        case DHCR_ADDRESS_WRAP:
          if (jy2<0)  jy2 = -jy2;
          if (jy>=h)  jy  =  h - 1 - (jy - h);
    
          if (jy2>=h) jy2 =  h - 1 - (jy2 - h);
          if (jy2<0)  jy2 = -jy2;
          break;
      }
      
      int j = ((jy * w) + tid.x);
      int j2 = ((jy2 * w) + tid.x);
      
      if (j>w*h) continue;
      if (j2>w*h) continue;
      
      val += scl[j] * weights[i+half_size] * f + scl[j2] * weights[i] * f;
    }
    
    tcl[index] = val;

  }
}

#endif //DEHANCER_GPULIB_UNARY_KERNELS_H
