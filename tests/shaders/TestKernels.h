//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_TESTKERNELS_H
#define DEHANCER_GPULIB_TESTKERNELS_H

#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__DEHANCER_KERNEL__ void kernel_vec_add(
        __DEHANCER_DEVICE_ARG__   float* A BIND_BUFFER(0),
        __DEHANCER_DEVICE_ARG__   float* B BIND_BUFFER(1),
        __DEHANCER_DEVICE_ARG__   float* C BIND_BUFFER(2),
        __DEHANCER_CONST_ARG__ __int_ref N BIND_BUFFER(3))
{
  int tid; get_kernel_tid1d(tid);
  if (tid < N)
  C[tid] = A[tid] + B[tid];
}

__DEHANCER_KERNEL__ void kernel_vec_dev(
        __DEHANCER_DEVICE_ARG__    float* C BIND_BUFFER(0),
        __DEHANCER_CONST_ARG__  __int_ref N BIND_BUFFER(1)
)
{
    int tid; get_kernel_tid1d(tid);
    if (tid < N)
    C[tid] /= 3.0f;
}

__DEHANCER_KERNEL__ void kernel_test_simple_transform(
        __read_only  image2d_t source       BIND_TEXTURE(0),
        __write_only image2d_t destination  BIND_TEXTURE(1)
) {
  // Calculate surface coordinates
  Texel2d tex; get_kernel_texel2d(destination,tex);

  if (!get_texel_boundary(tex)) return;

  float2 coords = get_texel_coords(tex);

  float4 color = sampled_color(source, destination, tex.gid); color.x = 0;

  write_image(destination, color, tex.gid);
}

__DEHANCER_KERNEL__  void kernel_make3DLut_transform(
                __write_only      image3d_t      d3DLut BIND_TEXTURE(0),
        __DEHANCER_CONST_ARG__ __float2_ref compression BIND_BUFFER(1)
)
{

    Texel3d tex; get_kernel_texel3d(d3DLut,tex);

    if (!get_texel_boundary(tex)) return;

    float3 c = compress(get_texel_coords(tex), compression);

    // transformation
    float4 color = (float4){c.x/2.f, c.y, 0.f, 1.f};

    write_image(d3DLut, color, tex.gid);
}

__DEHANCER_KERNEL__ void kernel_grid_test_transform(
        __read_only  image2d_t      source BIND_TEXTURE(0),
        __write_only image2d_t destination BIND_TEXTURE(1),
        __read_only  image3d_t      d3DLut BIND_TEXTURE(2),
        __read_only  image1d_t      d1DLut BIND_TEXTURE(3))
{
    // Calculate surface coordinates
    Texel2d tex; get_kernel_texel2d(destination,tex);

    if (!get_texel_boundary(tex)) return;

    float2 coords = get_texel_coords(tex);

    float4 color = sampled_color(source, destination, tex.gid);

    color = read_image(d3DLut, color);

    color = read_image(d1DLut, color);

    write_image(destination, color, tex.gid);
}

__DEHANCER_KERNEL__ void kernel_make1DLut_transform(
                __write_only     image1d_t  d1DLut      BIND_TEXTURE(0),
        __DEHANCER_CONST_ARG__ __float2_ref compression BIND_BUFFER(1))
{

    Texel1d tex; get_kernel_texel1d(d1DLut,tex);

    if (!get_texel_boundary(tex)) return;

    float3 denom = (float3){tex.size, tex.size, tex.size};

    float x = tex.gid;

    float3 c = compress((float3){x, x, x}/denom, compression);

    // linear transform with compression
    float4 color = (float4){c.x, c.y, c.z, 1.f};

    write_image(d1DLut, color, x);
}

__DEHANCER_KERNEL__ void ao_bench_kernel(
        __DEHANCER_CONST_ARG__ __int_ref nsubsamples  BIND_BUFFER(0),
                __write_only   image2d_t destination BIND_TEXTURE(1)
)
{

    Texel2d tex; get_kernel_texel2d(destination,tex);

    if (!get_texel_boundary(tex)) return;

    float4 color = ao_bench(nsubsamples, tex.gid.x, tex.gid.y, tex.size.x, tex.size.y);

    write_image(destination, color, tex.gid);
}

__DEHANCER_KERNEL__ void blend_kernel(
                  __read_only     image2d_t      source BIND_TEXTURE(0),
                  __write_only    image2d_t destination BIND_TEXTURE(1),
          __DEHANCER_DEVICE_ARG__    float*   color_map BIND_BUFFER(2),
        __DEHANCER_CONST_ARG__    __int_ref      levels BIND_BUFFER(3),
        __DEHANCER_CONST_ARG__ __float3_ref     opacity BIND_BUFFER(4)
) {
    Texel2d tex; get_kernel_texel2d(destination,tex);

    if (!get_texel_boundary(tex)) return;

    float2 coords = get_texel_coords(tex);

    float4 inColor = read_image(source, coords);

    float3        c = (float3){inColor.x,inColor.y,inColor.z};
    float luminance = dot(c, kIMP_Y_YUV_factor);
    int       index = clamp((int)(luminance*(float)(levels-1)),(int)(0),(int)(levels-1));
    float4    color = {1.0, 0.0, 0.0, 1.0};

    if (index<levels){
      color.x = color_map[index*3];
      color.y = color_map[index*3+1];
      color.z = color_map[index*3+2];
    }

    float3  r = (float3){color.x,color.y,color.z};

    float3 rc = mix(c,r,opacity);

    color.x = rc.x; color.y = rc.y; color.z = rc.z;

    write_image(destination, color, tex.gid);
}

#endif //DEHANCER_GPULIB_TESTKERNELS_H
