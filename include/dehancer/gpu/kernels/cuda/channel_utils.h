//
// Created by denn nevera on 30/11/2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/common.h"
#include "dehancer/gpu/kernels/cuda/std_kernels.h"
#include "dehancer/gpu/kernels/resample.h"

extern "C" __global__ void swap_channels_kernel (
        float* scl,
        float* tcl,
        int w,
        int h)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    tcl[index] = scl[index];
  }
}

extern "C" __global__ void image_to_channels (
        dehancer::nvcc::texture2d<float4> source,
        float* reds,
        float* greens,
        float* blues,
        float* alphas
        ,
        float4_ref_t slope
        ,
        float4_ref_t offset
        ,
        bool4_ref_t transform
        ,
        TransformDirection direction
        ,
        bool_ref_t has_mask
        ,
        texture2d_read_t mask
)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = source.get_width();
  int h = source.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    int2 destination_size = make_int2(w,h);
    
    float4 color     = sampled_color(source, destination_size, gid);
    
    float4  eColor = has_mask ? sampled_color(mask, destination_size, gid) : make_float4(1.0f);
    
    if (transform.x)
      color.x = linearlog( color.x, slope.x, offset.x, direction, eColor.x);
    
    if (transform.y)
      color.y = linearlog( color.y, slope.y, offset.y, direction, eColor.y);
    
    if (transform.z)
      color.z = linearlog( color.z, slope.z, offset.z, direction, eColor.z);
    
    if (transform.w)
      color.w = linearlog( color.w, slope.w, offset.w, direction, eColor.w);
    
    reds[index]   = color.x;
    greens[index] = color.y;
    blues[index]  = color.z;
    alphas[index] = color.w;
  }
}

extern "C" __global__ void channels_to_image (
        dehancer::nvcc::texture2d<float4> destination,
        float* reds,
        float* greens,
        float* blues,
        float* alphas
        ,
        float4_ref_t slope
        ,
        float4_ref_t offset
        ,
        bool4_ref_t transform
        ,
        TransformDirection direction
        ,
        bool_ref_t has_mask
        ,
        texture2d_read_t mask
)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = destination.get_width();
  int h = destination.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h)) {
    
    const int index = ((gid.y * w) + gid.x);
    int2 destination_size = make_int2(w,h);
    
    float4 color = make_float4(reds[index], greens[index], blues[index], alphas[index]);
    
    float4  eColor = has_mask ? sampled_color(mask, destination_size, gid) : make_float4(1.0f);
    
    if (transform.x)
      color.x = linearlog( color.x, slope.x, offset.x, direction, eColor.x);
    
    if (transform.y)
      color.y = linearlog( color.y, slope.y, offset.y, direction, eColor.y);
    
    if (transform.z)
      color.z = linearlog( color.z, slope.z, offset.z, direction, eColor.z);
    
    if (transform.w)
      color.w = linearlog( color.w, slope.w, offset.w, direction, eColor.w);
    
    write_image(destination, color, gid);
  }
}

typedef union {
    float4 vec;
    float  arr[4];
} _channel_tr_;

extern "C" __global__ void image_to_one_channel (
        dehancer::nvcc::texture2d<float4> source
        ,
        float* channel
        ,
        int channel_w
        ,
        int channel_h
        ,
        int    channel_index
        ,
        float_ref_t slope
        ,
        float_ref_t offset
        ,
        bool_ref_t transform
        ,
        TransformDirection direction
        ,
        bool_ref_t has_mask
        ,
        texture2d_read_t mask
)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = source.get_width();
  int h = source.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < channel_w) && (gid.y < channel_h) && channel_index<4) {
    
    const int index = ((gid.y * channel_w) + gid.x);
    
    int2 size = make_int2(channel_w,channel_h);
    
    _channel_tr_ color; color.vec = bicubic_sampled_color(source, size, gid);

    _channel_tr_ eColor; eColor.vec = has_mask ? sampled_color(mask, size, gid) : make_float4(1.0f);

    if (transform)
      color.arr[channel_index] = linearlog( color.arr[channel_index], slope, offset, direction, eColor.arr[channel_index]);

    channel[index]   = color.arr[channel_index];
  }
}

extern "C" __global__ void one_channel_to_image (
        dehancer::nvcc::texture2d<float4> source
        ,
        dehancer::nvcc::texture2d<float4> destination
        ,
        float* channel
        ,
        int channel_w
        ,
        int channel_h
        ,
        int    channel_index
        ,
        float_ref_t slope
        ,
        float_ref_t offset
        ,
        bool_ref_t transform
        ,
        TransformDirection direction
        ,
        bool_ref_t has_mask
        ,
        texture2d_read_t mask
)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int w = destination.get_width();
  int h = destination.get_height();
  
  int2 gid = (int2){x, y};
  
  if ((gid.x < w) && (gid.y < h) && channel_index<4) {
    
    int2 destination_size = make_int2(w,h);
    
    _channel_tr_ color; color.vec = sampled_color(source, destination_size, gid);
    
    float2 scale  = make_float2((float)channel_w,(float)channel_h)/make_float2((float)w,(float)h);
    float2 coords = make_float2((float)x, (float)y) * scale;
    
    int2 size = make_int2(channel_w,channel_h);
  
    if (size.x==w && size.y==h) {
      const int index = ((gid.y * channel_w) + gid.x);
      color.arr[channel_index] = channel[index];
    }
    else {
      color.arr[channel_index] = channel_bicubic(channel, size, coords.x, coords.y);
    }

    _channel_tr_ eColor; eColor.vec = has_mask ? sampled_color(mask, destination_size, gid) : make_float4(1.0f);

    if (transform)
      color.arr[channel_index] = linearlog( color.arr[channel_index], slope, offset, direction, eColor.arr[channel_index]);

    write_image(destination, color.vec, gid);
  }
}
