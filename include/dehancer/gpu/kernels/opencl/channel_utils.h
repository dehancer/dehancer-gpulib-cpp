//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_CHANNEL_UTILS_H
#define DEHANCER_GPULIB_CHANNEL_UTILS_H

__kernel void image_to_channels (
        __read_only image2d_t source,
        __global float* reds,
        __global float* greens,
        __global float* blues,
        __global float* alphas)
{
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(source);
  int h = get_image_height(source);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);

    float4 color     = read_imagef(source, sampler, gid);

    reds[index] = color.r;
    greens[index] = color.g;
    blues[index] = color.b;
    alphas[index] = color.rgba.a;
  }

}

__kernel void channels_to_image (
        __write_only image2d_t destination,
        __global float* reds,
        __global float* greens,
        __global float* blues,
        __global float* alphas)
{
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(destination);
  int h = get_image_height(destination);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    float4 inColor = {reds[index], greens[index], blues[index], alphas[index]};
    write_imagef(destination, gid, inColor);
  }
}

#endif //DEHANCER_GPULIB_CHANNEL_UTILS_H
