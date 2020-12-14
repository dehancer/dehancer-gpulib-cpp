//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/kernels/opencl/channel_utils.h"
#include "dehancer/gpu/kernels/opencl/blur_kernels.h"
#include "dehancer/gpu/kernels/opencl/common.h"

#include "aoBenchKernel.h"

__kernel void ao_bench_kernel(int nsubsamples, __write_only image2d_t destination )
{

  int w = get_image_width (destination);
  int h = get_image_height (destination);

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float4 color = ao_bench(nsubsamples, x, y, w, h);

  write_imagef(destination, gid, color);

}


__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__kernel void blend_kernel(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        __global float* color_map,
        uint levels,
        float3 opacity
) {

  int2 gid = (int2)(get_global_id(0),
                    get_global_id(1));

  int2 imageSize = (int2)(get_image_width(destination),
                          get_image_height(destination));

  if (gid.x >= imageSize.x || gid.y >= imageSize.y)
  {
    return;
  }

  // Normalize coordinates
  float2 coords = (float2)((float)gid.x / (imageSize.x - 1),
                           (float)gid.y / (imageSize.y - 1));


  float4 inColor = read_imagef(source, sampler, coords);

  float luminance = dot(inColor.xyz, kIMP_Y_YUV_factor);
  int      index = clamp((int)(luminance*(float)(levels-1)),(int)(0),(int)(levels-1));
  float4   color = {1.0, 0.0, 0.0, 1.0};

  if (index<levels){
    color.x = color_map[index*3];
    color.y = color_map[index*3+1];
    color.z = color_map[index*3+2];
  }

  color.xyz = mix(inColor.xyz,color.xyz,opacity);

  write_imagef(destination, gid, color);
}

__kernel void convolve_horizontal_image_kernel(
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

__kernel void convolve_vertical_kernel (__global float* scl,
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
