//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/kernels/opencl/opencl.h"
#include "aoBenchKernel.h"
#include "TestKernels.h"


__DEHANCER_KERNEL__ void convolve_horizontal_image_kernel(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        __global float* weights,
        int size
) {

  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(source);
  int h = get_image_height(source);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    float4 color = (float4)(0,0,0,1);
    for (int i = -size/2; i < size/2; ++i) {
      int2 gidx = gid;
      gidx.x += i;
      if (gidx.x<0) continue;
      if (gidx.x>=w) continue;
      color += read_imagef(source, nearest_sampler, gidx) * weights[i];
    }
    write_imagef(destination, gid, color);
  }

}

__kernel void convolve_horizontal_kernel (__global float* scl,
                                          __global float* tcl,
                                          int w,
                                          int h,
                                          __global float* weights,
                                          int size
) {

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float val = 0;
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    for (int i = -size/2; i < size/2; ++i) {
      int jx =  gid.x+i;
      if (jx<0) jx -= i;
      if (jx>=w) jx -= i;
      const int j = ((gid.y * w) + jx);
      val += scl[j] * weights[i+size/2];
    }
    tcl[index] = val;
  }
}

__DEHANCER_KERNEL__ void convolve_vertical_kernel (__global float* scl,
                                        __global float* tcl,
                                        int w,
                                        int h,
                                        __global float* weights,
                                        int size
) {

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float val = 0;
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    for (int i = -size/2; i < size/2; ++i) {
      int jy =  gid.y+i;
      if (jy<0) jy -= i;
      if (jy>=h) jy -= i;
      const int j = ((jy * w) + gid.x);
      val += scl[j] * weights[i+size/2];
    }
    tcl[index] = val;
  }
}
