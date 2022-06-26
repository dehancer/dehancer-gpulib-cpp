//
// Created by denn on 22.06.2022.
//

#ifndef DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H
#define DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/histogram_common.h"

DHCR_KERNEL void kernel_histogram_image(
        texture2d_read_t       img       DHCR_BIND_TEXTURE(0),
        DHCR_DEVICE_ARG uint*  partial_histogram DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG  int_ref_t num_groups DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_2D
) {
  
  Texel2d tex; get_kernel_texel2d(img,tex);
  if (!get_texel_boundary(tex)) return;
  
  int     image_width  = tex.size.x;
  int     image_height = tex.size.y;
  int     x = tex.gid.x;
  int     y = tex.gid.y;
  
  uint    num_blocks = (uint)get_num_blocks_2d();
  int2    block_size = get_block_size2d();
  int     local_size = block_size.x * block_size.y;
  
  uint2    block_id = (uint2)get_block_id2d();
  int    group_indx = (int)mad24( block_id.y, num_blocks, block_id.x) * DEHANCER_HISTOGRAM_BUFF_LENGTH;
  
  DHCR_BLOCK_MEMORY atomic_uint tmp_histogram[DEHANCER_HISTOGRAM_BUFF_LENGTH];
  
  uint2   thread_in_block_id = (uint2)get_thread_in_block_id2d();
  int     tid = mad24(thread_in_block_id.y, (uint)block_size.x, thread_in_block_id.x);
  
  // clear the local buffer that will generate the partial histogram
  #pragma unroll
  for (int j = DEHANCER_HISTOGRAM_BUFF_LENGTH, indx = 0; j >0 ; j -= local_size) {
    if (tid < j){
      atomic_store(tmp_histogram[indx+tid],0);
    }
    indx += local_size;
  }
  
  block_barrier();
  
  if ((x < image_width) && (y < image_height))
  {
    float4 clr = read_image(img, make_int2(x,y));
    
    /**
     * #pragma unroll DEHANCER_HISTOGRAM_CHANNELS
     * for (;i<DEHANCER_HISTOGRAM_CHANNELS;)
     */
    uint   nindx = (uint)(min(clr.x, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    atomic_fetch_inc(tmp_histogram[0*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx]);
    
    nindx = (uint)(min(clr.y, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    atomic_fetch_inc(tmp_histogram[1*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx]);
    
    nindx = (uint)(min(clr.z, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    atomic_fetch_inc(tmp_histogram[2*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx]);
  
    float3        c = make_float3(clr.x, clr.y, clr.z);
    float luminance = dot(c, kIMP_Y_YUV_factor);
  
    nindx = (uint)(min(luminance, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    atomic_fetch_inc(tmp_histogram[3*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx]);
  }
  
  block_barrier();
  
  // copy the partial histogram to appropriate location in histogram given by group_indx
  if (local_size >= (DEHANCER_HISTOGRAM_BUFF_LENGTH))
  {
    if (tid < (DEHANCER_HISTOGRAM_BUFF_LENGTH))
      partial_histogram[group_indx + tid] = atomic_load(tmp_histogram[tid]);
  }
  else
  {
    #pragma unroll
    for (int j = DEHANCER_HISTOGRAM_BUFF_LENGTH, indx = 0; j >0 ; j -= local_size) {
      
      if (tid < j)
        partial_histogram[group_indx + indx + tid] =  atomic_load(tmp_histogram[indx + tid]);
      indx += local_size;
    }
  }
}

DHCR_KERNEL void kernel_sum_partial_histogram_image(
        DHCR_DEVICE_ARG uint*     partial_histogram DHCR_BIND_BUFFER(0),
        DHCR_CONST_ARG  int_ref_t num_groups DHCR_BIND_BUFFER(1),
        DHCR_DEVICE_ARG uint*     histogram DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_1D
) {
  
  int tid = get_thread_in_grid_id1d();
  int block_id = get_block_id1d();
  
  DHCR_BLOCK_MEMORY uint tmp_histogram[DEHANCER_HISTOGRAM_BUFF_LENGTH];
  
  int thread_in_block_id = get_thread_in_block_id1d();
  int first_workitem_not_in_first_group = ((thread_in_block_id == 0) && block_id);
  
  tid += block_id;
  int     tid_first = tid - 1;
  if (first_workitem_not_in_first_group)
    tmp_histogram[tid_first] = partial_histogram[tid_first];
  
  tmp_histogram[tid] = partial_histogram[tid];
  
  int block_indx = DEHANCER_HISTOGRAM_BUFF_LENGTH;
  #pragma unroll
  for (int n = num_groups-1; n > 0; n--)
  {
    if (first_workitem_not_in_first_group)
      tmp_histogram[tid_first] += partial_histogram[tid_first];
    
    tmp_histogram[tid] += partial_histogram[block_indx+tid];
    block_indx += DEHANCER_HISTOGRAM_BUFF_LENGTH;
  }
  
  block_barrier();
  
  if (first_workitem_not_in_first_group)
    histogram[tid_first] = tmp_histogram[tid_first];
  
  histogram[tid] = tmp_histogram[tid];
}

#endif //DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H
