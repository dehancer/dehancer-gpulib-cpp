//
// Created by denn on 09.01.2021.
//

#ifndef DEHANCER_GPULIB_UNARY_KERNELS_H
#define DEHANCER_GPULIB_UNARY_KERNELS_H

#include "dehancer/gpu/kernels/types.h"

__DEHANCER_KERNEL__ void convolve_horizontal_kernel (
        __DEHANCER_DEVICE_ARG__     float*       scl BIND_BUFFER(0),
        __DEHANCER_DEVICE_ARG__     float*       tcl BIND_BUFFER(1),
        __DEHANCER_CONST_ARG__    __int_ref        w BIND_BUFFER(2),
        __DEHANCER_CONST_ARG__    __int_ref        h BIND_BUFFER(3),
        __DEHANCER_DEVICE_ARG__       float* weights BIND_BUFFER(4),
        __DEHANCER_CONST_ARG__    __int_ref     size BIND_BUFFER(5),
        __DEHANCER_CONST_ARG__    __int_ref  address BIND_BUFFER(6)
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
      switch ((EdgeAddress)address) {
        case ADDRESS_CLAMP:
          if (jx<0)  jx = 0;
          if (jx>=w) jx = w-1;
          break;
        case ADDRESS_BORDER:
          if (jx<0)  {jx = 0;f = 0;}
          if (jx>=w) {jx = w-1;f = 0;}
          break;
        case ADDRESS_WRAP:
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

__DEHANCER_KERNEL__ void convolve_vertical_kernel (
        __DEHANCER_DEVICE_ARG__     float*       scl BIND_BUFFER(0),
        __DEHANCER_DEVICE_ARG__     float*       tcl BIND_BUFFER(1),
        __DEHANCER_CONST_ARG__    __int_ref        w BIND_BUFFER(2),
        __DEHANCER_CONST_ARG__    __int_ref        h BIND_BUFFER(3),
        __DEHANCER_DEVICE_ARG__      float*  weights BIND_BUFFER(4),
        __DEHANCER_CONST_ARG__    __int_ref     size BIND_BUFFER(5),
        __DEHANCER_CONST_ARG__    __int_ref  address BIND_BUFFER(6)
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
      switch ((EdgeAddress)address) {
        case ADDRESS_CLAMP:
          if (jy<0)  jy = 0;
          if (jy>=h) jy = h-1;
          break;
        case ADDRESS_BORDER:
          if (jy<0)  {jy = 0;f = 0;}
          if (jy>=h) {jy = h-1;f = 0;}
          break;
        case ADDRESS_WRAP:
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
