//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_BLUR_KERNELS_H
#define DEHANCER_GPULIB_BLUR_KERNELS_H


__kernel void box_blur_horizontal_kernel (__global float* scl,
                                          __global float* tcl,
                                          int w,
                                          int h,
                                          int r) {
  float iarr = 1.0f / (float)(r+r+1);
  int i = get_global_id(1);
  if (i>=h) return ;
  int ti = i*w, li = ti, ri = ti+r;
  float fv = scl[ti], lv = scl[ti+w-1], val = (float)(r+1)*fv;
  for(int j=0; j<r; j++) val += scl[ti+j];
  for(int j=0  ; j<=r ; j++) { val += scl[ri++] - fv       ;   tcl[ti++] = val*iarr; }
  for(int j=r+1; j<w-r; j++) { val += scl[ri++] - scl[li++];   tcl[ti++] = val*iarr; }
  for(int j=w-r; j<w  ; j++) { val += lv        - scl[li++];   tcl[ti++] = val*iarr; }
}

__kernel void box_blur_vertical_kernel (__global float* scl,
                                        __global float* tcl,
                                        int w,
                                        int h,
                                        int r) {
  float iarr = 1.0f / (float)(r+r+1);
  int i = get_global_id(0);
  if (i>=w) return ;
  int ti = i, li = ti, ri = ti+r*(float)w;
  float fv = scl[ti], lv = scl[ti+w*(h-1)], val = (float )(r+1)*fv;
  for(int j=0; j<r; j++) val += scl[ti+j*w];
  for(int j=0  ; j<=r ; j++) { val += scl[ri] - fv     ;  tcl[ti] = val*iarr;  ri+=w; ti+=w; }
  for(int j=r+1; j<h-r; j++) { val += scl[ri] - scl[li];  tcl[ti] = val*iarr;  li+=w; ri+=w; ti+=w; }
  for(int j=h-r; j<h  ; j++) { val += lv      - scl[li];  tcl[ti] = val*iarr;  li+=w; ti+=w; }
}

#endif //DEHANCER_GPULIB_BLUR_KERNELS_H
