//
// Created by denn on 22.06.2022.
//

#ifndef DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H
#define DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/histogram_common.h"

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define DEHANCER_HISTOGRAM_BUFF_SIZE (DEHANCER_HISTOGRAM_SIZE+1)
#define DEHANCER_HISTOGRAM_BUFF_LENGTH (DEHANCER_HISTOGRAM_BUFF_SIZE * DEHANCER_HISTOGRAM_CHANNELS)
#define DEHANCER_HISTOGRAM_MULT ((float)(DEHANCER_HISTOGRAM_SIZE))

DHCR_KERNEL void kernel_histogram_image(
        texture2d_read_t       img       DHCR_BIND_TEXTURE(0),
        DHCR_DEVICE_ARG int*    histogram DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_2D
) {
  
  int     local_size = (int)get_local_size(0) * (int)get_local_size(1);
  int     image_width = get_image_width(img);
  int     image_height = get_image_height(img);
  int     group_indx = (int)mad24((uint)get_group_id(1), (uint)get_num_groups(0), (uint)get_group_id(0)) * 257 * 3;
  int     x = get_global_id(0);
  int     y = get_global_id(1);
  
  local uint  tmp_histogram[DEHANCER_HISTOGRAM_BUFF_LENGTH];
  
  int     tid = mad24((uint)get_local_id(1), (uint)get_local_size(0), (uint)get_local_id(0));
  int     j = DEHANCER_HISTOGRAM_BUFF_LENGTH;
  int     indx = 0;
  
  // clear the local buffer that will generate the partial histogram
  do
  {
    if (tid < j)
      tmp_histogram[indx+tid] = 0;
    
    j -= local_size;
    indx += local_size;
  } while (j > 0);
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if ((x < image_width) && (y < image_height))
  {
    float4 clr = read_imagef(img, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (float2)(x, y));
    
    ushort   indx;
    indx = convert_ushort_sat(min(clr.x, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    atom_inc(&tmp_histogram[indx]);
    
    indx = convert_ushort_sat(min(clr.y, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    atom_inc(&tmp_histogram[DEHANCER_HISTOGRAM_BUFF_SIZE+indx]);
    
    indx = convert_ushort_sat(min(clr.z, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    atom_inc(&tmp_histogram[2*DEHANCER_HISTOGRAM_BUFF_SIZE+indx]);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // copy the partial histogram to appropriate location in histogram given by group_indx
  if (local_size >= (DEHANCER_HISTOGRAM_BUFF_LENGTH))
  {
    if (tid < (DEHANCER_HISTOGRAM_BUFF_LENGTH))
      histogram[group_indx + tid] = tmp_histogram[tid];
  }
  else
  {
    j = DEHANCER_HISTOGRAM_BUFF_LENGTH;
    indx = 0;
    do
    {
      
      if (tid < j)
        histogram[group_indx + indx + tid] = tmp_histogram[indx + tid];
      
      j -= local_size;
      indx += local_size;
    } while (j > 0);
  }
}

#endif //DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H
