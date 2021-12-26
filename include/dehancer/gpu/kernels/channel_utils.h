//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_CHANNEL_UTILS_LIB_H
#define DEHANCER_GPULIB_CHANNEL_UTILS_LIB_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

DHCR_KERNEL void swap_channels_kernel (
        DHCR_DEVICE_ARG float* scl  DHCR_BIND_BUFFER(0),
        DHCR_DEVICE_ARG float* tcl  DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG int_ref_t w  DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG int_ref_t h  DHCR_BIND_BUFFER(3)
        
        DHCR_KERNEL_GID_2D
)
{
  int2 gid; get_kernel_tid2d(gid);
  
  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    tcl[index] = scl[index];
  }
}

typedef union {
    float4 vec;
    float  arr[4];
} channel_tr_;

DHCR_KERNEL void image_to_one_channel (
        texture2d_read_t                source DHCR_BIND_TEXTURE(0),
        DHCR_DEVICE_ARG float*         channel DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG int_ref_t     channel_w DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG int_ref_t     channel_h DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG int_ref_t channel_index DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG float_ref_t       slope DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG float_ref_t      offset DHCR_BIND_BUFFER(6),
        DHCR_CONST_ARG bool_ref_t    transform DHCR_BIND_BUFFER(7),
        DHCR_CONST_ARG_REF (DHCR_TransformDirection) direction DHCR_BIND_BUFFER(8),
        DHCR_CONST_ARG_REF (DHCR_TransformType) trtype DHCR_BIND_BUFFER(9)//,
//        DHCR_CONST_ARG bool_ref_t     has_mask DHCR_BIND_BUFFER(10),
//        texture2d_read_t                  mask DHCR_BIND_TEXTURE(11)
        
        DHCR_KERNEL_GID_2D
)
{
  int2 gid; get_kernel_tid2d(gid);
  
  if ((gid.x < channel_w) && (gid.y < channel_h) && channel_index<4) {
    
    const int index = ((gid.y * channel_w) + gid.x);
    
    int2 size = make_int2(channel_w,channel_h);
    
    channel_tr_ color; color.vec = sampled_color(source, size, gid);

    //channel_tr_ eColor; eColor.vec = has_mask ? sampled_color(mask, size, gid) : to_float4(1.0f);
    channel_tr_ eColor; eColor.vec = to_float4(1.0f);

    if (transform)
      switch (trtype) {
        case DHCR_log_linear:
          color.arr[channel_index] = linear_log(color.arr[channel_index], slope, offset, direction,
                                                eColor.arr[channel_index]);
          break;
        case DHCR_pow_linear:
          color.arr[channel_index] = linear_pow(color.arr[channel_index], slope, offset, direction,
                                                eColor.arr[channel_index]);
          break;
      }
    
    channel[index]   = color.arr[channel_index];
  }
}

DHCR_KERNEL void one_channel_to_image (
        texture2d_read_t                source DHCR_BIND_TEXTURE(0),
        texture2d_write_t          destination DHCR_BIND_TEXTURE(1),
        DHCR_DEVICE_ARG float*         channel DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG int_ref_t     channel_w DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG int_ref_t     channel_h DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG int_ref_t channel_index DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG float_ref_t       slope DHCR_BIND_BUFFER(6),
        DHCR_CONST_ARG float_ref_t      offset DHCR_BIND_BUFFER(7),
        DHCR_CONST_ARG bool_ref_t    transform DHCR_BIND_BUFFER(8),
        DHCR_CONST_ARG_REF (DHCR_TransformDirection) direction DHCR_BIND_BUFFER(9),
        DHCR_CONST_ARG_REF (DHCR_TransformType) trtype DHCR_BIND_BUFFER(10)//,
//        DHCR_CONST_ARG bool_ref_t     has_mask DHCR_BIND_BUFFER(11),
//        texture2d_read_t                  mask DHCR_BIND_TEXTURE(12)//,
        //DHCR_CONST_ARG bool_ref_t     has_channel DHCR_BIND_BUFFER(13)

        DHCR_KERNEL_GID_2D

)
{
  int2 gid; get_kernel_tid2d(gid);
  
  int w = get_texture_width(destination);
  int h = get_texture_height(destination);
  
  if ((gid.x < w) && (gid.y < h) && channel_index<4) {
    
    int2 destination_size = make_int2(w,h);
    
    channel_tr_ color; color.vec = sampled_color(source, destination_size, gid);
    
    //if (!has_channel) {
    //  write_image(destination, color.vec, gid);
    //  return;
    //}
    
    float2 scale  = make_float2((float)channel_w,(float)channel_h)/make_float2((float)w,(float)h);
    float2 coords = make_float2((float)gid.x, (float)gid.y) * scale;
    
    int2 size = make_int2(channel_w,channel_h);
  
    if (size.x==w && size.y==h) {
      const int index = ((gid.y * channel_w) + gid.x);
      color.arr[channel_index] = channel[index];
    }
    else {
      color.arr[channel_index] = channel_bicubic(channel, size, coords.x, coords.y);
    }

    //channel_tr_ eColor; eColor.vec = has_mask ? sampled_color(mask, destination_size, gid) : to_float4(1.0f);
    channel_tr_ eColor; eColor.vec = to_float4(1.0f);
  
    if (transform)
      switch (trtype) {
        case DHCR_log_linear:
          color.arr[channel_index] = linear_log( color.arr[channel_index], slope, offset, direction, eColor.arr[channel_index]);
          break;
        case DHCR_pow_linear:
          color.arr[channel_index] = linear_pow( color.arr[channel_index], slope, offset, direction, eColor.arr[channel_index]);
          break;
      }

    write_image(destination, color.vec, gid);
  }
}

#endif // DEHANCER_GPULIB_CHANNEL_UTILS_LIB_H