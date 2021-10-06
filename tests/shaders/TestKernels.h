//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_TEST_KERNELS_CPP
#define DEHANCER_GPULIB_TEST_KERNELS_CPP

#include "test_struct.h"
#include "dehancer/gpu/kernels/lib.h"
#include "aoBenchKernel.h"

DHCR_KERNEL void kernel_vec_add(
        DHCR_DEVICE_ARG   float* A DHCR_BIND_BUFFER(0) ,
        DHCR_DEVICE_ARG   float* B DHCR_BIND_BUFFER(1) ,
        DHCR_DEVICE_ARG   float* C DHCR_BIND_BUFFER(2) ,
        DHCR_CONST_ARG int_ref_t N DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG_REF (TestStruct) data DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG_REF (DHCR_LogParameters) data2 DHCR_BIND_BUFFER(5)
        DHCR_KERNEL_GID_1D
)
{
  float3x3 m =
          (float3x3){
                  3.90405e-1f, 5.49941e-1f, 8.92632e-3f,
                  7.08416e-2f, 9.63172e-1f, 1.35775e-3f,
                  2.31082e-2f, 1.28021e-1f, 9.36245e-1f
          };
          
  float3 cc = to_float3(1);
  
  cc = matrix_mul(m,cc);
  
  DHCR_LogParameters d=data2;
  int tid; get_kernel_tid1d(tid);
  if (tid < N)
    C[tid] = A[tid] + B[tid] + data.data*data.size;
}

DHCR_KERNEL void kernel_vec_dev(
        DHCR_DEVICE_ARG    float* C DHCR_BIND_BUFFER(0),
        DHCR_CONST_ARG  int_ref_t N DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_1D
)
{
  int tid; get_kernel_tid1d(tid);
  if (tid < N)
    C[tid] /= 3.0f;
}

DHCR_KERNEL void kernel_test_simple_transform(
        texture2d_read_t source       DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination  DHCR_BIND_TEXTURE(1)
        DHCR_KERNEL_GID_2D
)
{
  // Calculate surface coordinates
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4 color = sampled_color(source, tex.size, tex.gid); color.x = 0;
  
  write_image(destination, color, tex.gid);
}

DHCR_KERNEL void kernel_make1DLut_transform(
        texture1d_write_t  d1DLut      DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t compression DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_1D
)
{
  
  Texel1d tex; get_kernel_texel1d(d1DLut,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float3 denom = make_float3(tex.size, tex.size, tex.size);
  
  float x = (float)(tex.gid);
  
  float3 c = compress(make_float3(x, x, x)/denom, compression);
  
  // linear transform with compression
  float4 color = make_float4(c.x, c.y, c.z, 1.f);
  
  write_image(d1DLut, color, tex.gid);
}

DHCR_KERNEL  void kernel_make3DLut_transform(
        texture3d_write_t      d3DLut DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t compression DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_3D
)
{
  
  Texel3d tex; get_kernel_texel3d(d3DLut,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float3 c = compress(get_texel_coords(tex), compression);
  
  // transformation
  float4 color = make_float4(c.x/2.f, c.y, 0.f, 1.f);
  
  write_image(d3DLut, color, tex.gid);
}

DHCR_KERNEL void kernel_test_transform(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        texture3d_read_t       d3DLut DHCR_BIND_TEXTURE(2),
        texture1d_read_t       d1DLut DHCR_BIND_TEXTURE(3)
        DHCR_KERNEL_GID_2D
        )
{
  // Calculate surface coordinates
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4 color = sampled_color(source, tex.size, tex.gid);
  
  color = read_image(d3DLut, color);
  
  color = read_image(d1DLut, color);
  
  write_image(destination, color, tex.gid);
}


DHCR_KERNEL void ao_bench_kernel(
        DHCR_CONST_ARG int_ref_t nsubsamples DHCR_BIND_BUFFER(0),
        texture2d_write_t        destination DHCR_BIND_TEXTURE(1)
        DHCR_KERNEL_GID_2D
)
{
  
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4 color = ao_bench(nsubsamples, tex.gid.x, tex.gid.y, tex.size.x, tex.size.y);
  
  write_image(destination, color, tex.gid);
}

DHCR_KERNEL void blend_kernel(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1),
        DHCR_DEVICE_ARG    float*   color_map DHCR_BIND_BUFFER(2),
        DHCR_CONST_ARG    int_ref_t      levels DHCR_BIND_BUFFER(3),
        DHCR_CONST_ARG float3_ref_t     opacity DHCR_BIND_BUFFER(4)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex; get_kernel_texel2d(destination,tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float2 coords = get_texel_coords(tex);
  
  float4 inColor = read_image(source, coords);
  
  float3        c = make_float3(inColor.x,inColor.y,inColor.z);
  float luminance = dot(c, kIMP_Y_YUV_factor);
  int       index = clamp((int)(luminance*(float)(levels-1)),(int)(0),(int)(levels-1));
  float4    color = {1.0f, 0.0f, 0.0f, 1.0f};
  
  if (index<levels){
    color.x = color_map[index*3];
    color.y = color_map[index*3+1];
    color.z = color_map[index*3+2];
  }
  
  float3  r = make_float3(color.x,color.y,color.z);
  
  float3 rc = mix(c,r,opacity);
  
  color.x = rc.x; color.y = rc.y; color.z = rc.z;
  
  write_image(destination, color, tex.gid);
}


typedef union {
    float4 vec;
    float a[4];
} U4;

DHCR_KERNEL void kernel_fast_convolve(
        texture2d_read_t           source DHCR_BIND_TEXTURE(0),
        texture2d_write_t     destination DHCR_BIND_TEXTURE(1),
        DHCR_DEVICE_ARG       float*   weights_array DHCR_BIND_BUFFER(2),
        DHCR_DEVICE_ARG       float*   offsets_array DHCR_BIND_BUFFER(3),
        DHCR_DEVICE_ARG         int*      step_count DHCR_BIND_BUFFER(4),
        DHCR_CONST_ARG    int_ref_t         channels DHCR_BIND_BUFFER(5),
        DHCR_CONST_ARG float2_ref_t        direction DHCR_BIND_BUFFER(6)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex; get_kernel_texel2d(destination, tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float2 coords = get_texel_coords(tex);
  
  U4 result;
  float2 pixel_size = {direction.x/(float)tex.size.x,direction.y/(float)tex.size.y};
  
  int next_array_index = 0;
  
  #pragma unroll
  for (int j = 0; j < channels; ++j) {
    
    if (j>=4) return;
    
    result.a[j] = 0;
    
    U4 color;
    
    if (step_count[j] == 0) {
      color.vec = read_image(source, coords);
      result.a[j] = color.a[j];
    }
    else {
      
      #pragma unroll
      for (int i = 0; i < step_count[j]; ++i) {
        float2 coords_offset = offsets_array[next_array_index + i] * pixel_size;
        
        float2 xy = coords + coords_offset;
        
        color.vec = read_image(source, xy);
        
        xy = coords - coords_offset;
        color.vec += read_image(source, xy);
        
        result.a[j] += weights_array[next_array_index + i] * color.a[j];
      }
    }
    
    next_array_index += step_count[j];
  }
  
  write_image(destination, result.vec, tex.gid);
  
}

DHCR_KERNEL void kernel_gradient(
        texture2d_write_t     destination DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG bool_ref_t inverse DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_2D
        )
{
  
  Texel2d tex; get_kernel_texel2d(destination, tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float2 coords = get_texel_coords(tex);
  
  float x = coords.x;
  float4 color = make_float4(x, x, x, 1.0f) ;
  
  if (inverse)
    color = to_float4(1.0f) - color;
    
  write_image(destination, color, tex.gid);
  
}

#endif //DEHANCER_GPULIB_TEST_KERNELS_CPP
