//
// Created by denn on 22.06.2022.
//

#ifndef DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H
#define DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/histogram_common.h"

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

DHCR_KERNEL void kernel_histogram_image(
        texture2d_read_t       img       DHCR_BIND_TEXTURE(0),
        DHCR_DEVICE_ARG uint*  partial_histogram DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG  int_ref_t num_groups DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_2D
        ,
        uint2 __dehancer_compute_size_2d__ [[threads_per_grid]],
        uint2 __dehancer_grid_size_2d__ [[threadgroups_per_grid]],
        uint2 __dehancer_block_size_2d__ [[threads_per_threadgroup]],
        uint2 __dehancer_block_id_2d__ [[threadgroup_position_in_grid]],
        uint2 __dehancer_thread_in_block_id__ [[thread_position_in_threadgroup]]

) {
  
  Texel2d tex; get_kernel_texel2d(img,tex);
  if (!get_texel_boundary(tex)) return;
  
  int     image_width  = tex.size.x;
  int     image_height = tex.size.y;
  int     x = tex.gid.x;
  int     y = tex.gid.y;
  
  uint    num_blocks = __dehancer_grid_size_2d__.x;//__dehancer_grid_size_2d__.x * __dehancer_grid_size_2d__.y;//(uint)get_num_blocks();
  int2    block_size = int2(__dehancer_block_size_2d__.x,__dehancer_block_size_2d__.y);//get_block_size2d();
  int     local_size = block_size.x * block_size.y;
  
  uint2   block_id = uint2(__dehancer_block_id_2d__.x,__dehancer_block_id_2d__.y);//get_block_id2d();
  int     group_indx = (int)mad24( block_id.y, num_blocks, block_id.x) * DEHANCER_HISTOGRAM_BUFF_LENGTH;
  
  threadgroup atomic_uint tmp_histogram[DEHANCER_HISTOGRAM_BUFF_LENGTH];

  //int     tid = mad24((uint)get_local_id(1), (uint)get_local_size(0), (uint)get_local_id(0));
  int     tid = mad24((uint)__dehancer_thread_in_block_id__.y, (uint)__dehancer_block_size_2d__.x, (uint)__dehancer_thread_in_block_id__.x);
  
  // clear the local buffer that will generate the partial histogram
  #pragma unroll
  for (int j = DEHANCER_HISTOGRAM_BUFF_LENGTH, indx = 0; j >0 ; j -= local_size) {
    if (tid < j){
      //tmp_histogram[indx+tid] = 0;
      atomic_store_explicit(&tmp_histogram[indx+tid], 0, memory_order_relaxed);
    }
    indx += local_size;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  //barrier(CLK_LOCAL_MEM_FENCE);
  
  if ((x < image_width) && (y < image_height))
  {
    float4 clr = read_image(img, make_int2(x,y)); //read_imagef(img, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (float2)(x, y));
  
    float3        c = make_float3(clr.x, clr.y, clr.z);
    float luminance = dot(c, kIMP_Y_YUV_factor);
  
    /**
     * #pragma unroll DEHANCER_HISTOGRAM_CHANNELS
     * for (;i<DEHANCER_HISTOGRAM_CHANNELS;)
     */
    uint   nindx = (uint)(min(clr.x, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    //atom_inc(&tmp_histogram[nindx]);
    atomic_fetch_add_explicit(&tmp_histogram[nindx], 1, memory_order_relaxed);
  
    nindx = (uint)(min(clr.y, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    //atom_inc(&tmp_histogram[1*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx]);
    atomic_fetch_add_explicit(&tmp_histogram[1*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx], 1, memory_order_relaxed);

    nindx = (uint)(min(clr.z, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    //atom_inc(&tmp_histogram[2*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx]);
    atomic_fetch_add_explicit(&tmp_histogram[2*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx], 1, memory_order_relaxed);

    nindx = (uint)(min(luminance, 1.0f) * DEHANCER_HISTOGRAM_MULT);
    //atom_inc(&tmp_histogram[3*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx]);
    atomic_fetch_add_explicit(&tmp_histogram[3*DEHANCER_HISTOGRAM_BUFF_SIZE+nindx], 1, memory_order_relaxed);
  }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  //barrier(CLK_LOCAL_MEM_FENCE);
  
  // copy the partial histogram to appropriate location in histogram given by group_indx
  if (local_size >= (DEHANCER_HISTOGRAM_BUFF_LENGTH))
  {
    if (tid < (DEHANCER_HISTOGRAM_BUFF_LENGTH))
      partial_histogram[group_indx + tid] = atomic_load_explicit(&tmp_histogram[tid], memory_order_relaxed);//tmp_histogram[tid];
  }
  else
  {
    //#pragma unroll
    for (int j = DEHANCER_HISTOGRAM_BUFF_LENGTH, indx = 0; j >0 ; j -= local_size) {
      
      if (tid < j)
        partial_histogram[group_indx + indx + tid] =  atomic_load_explicit(&tmp_histogram[indx + tid], memory_order_relaxed);//tmp_histogram[indx + tid];
        
//      partial_histogram[0*DEHANCER_HISTOGRAM_BUFF_SIZE+0] = 255;
//      partial_histogram[1*DEHANCER_HISTOGRAM_BUFF_SIZE+0] = 255;
//      partial_histogram[2*DEHANCER_HISTOGRAM_BUFF_SIZE+0] = 255;
//      partial_histogram[3*DEHANCER_HISTOGRAM_BUFF_SIZE+0] = 253;
//
//      partial_histogram[0*DEHANCER_HISTOGRAM_BUFF_SIZE+255] = 255;
//      partial_histogram[1*DEHANCER_HISTOGRAM_BUFF_SIZE+255] = 255;
//      partial_histogram[2*DEHANCER_HISTOGRAM_BUFF_SIZE+255] = 255;
//      partial_histogram[3*DEHANCER_HISTOGRAM_BUFF_SIZE+255] = 253;
      
      indx += local_size;
    }
  }
}

DHCR_KERNEL void kernel_sum_partial_histogram_image(
        DHCR_DEVICE_ARG uint*     partial_histogram DHCR_BIND_BUFFER(0),
        DHCR_CONST_ARG  int_ref_t num_groups DHCR_BIND_BUFFER(1),
        DHCR_DEVICE_ARG uint*     histogram DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_1D
        ,
        uint __dehancer_compute_size__ [[threads_per_grid]],
        uint __dehancer_grid_size__ [[threadgroups_per_grid]],
        uint __dehancer_block_size_ [[threads_per_threadgroup]],
        uint __dehancer_block_id__ [[threadgroup_position_in_grid]],
        uint __dehancer_thread_in_block_id__ [[thread_position_in_threadgroup]]
        //DHCR_KERNEL_LOCAL_1D
) {
  
  int tid = (uint)get_thread_in_grid_id1d();
  int block_id = (int)__dehancer_block_id__; //(uint)get_block_id1d();
  
  threadgroup uint tmp_histogram[DEHANCER_HISTOGRAM_BUFF_LENGTH];
  
  int thread_in_block_id = (int)__dehancer_thread_in_block_id__;//get_thread_in_block_id1d();
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
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  if (first_workitem_not_in_first_group)
    histogram[tid_first] = tmp_histogram[tid_first];
  
  histogram[tid] = tmp_histogram[tid];
}

#endif //DEHANCER_GPULIB_HISTOGRAM_IMAGE_KERNEL_H
