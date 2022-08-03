//
// Created by denn on 24.04.2021.
//

#ifndef DEHANCER_GPULIB_CLUT_KERNELS_NEW_H
#define DEHANCER_GPULIB_CLUT_KERNELS_NEW_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

//
// MARK - Identity
//

DHCR_KERNEL void kernel_make1DLut(
        texture1d_write_t         d1DLut DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t  compression DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_1D
) {
  Texel1d tex;
  get_kernel_texel1d(d1DLut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 denom = to_float3(get_texture_width(d1DLut)-1);
  float4 input_color  = to_float4(compress(to_float3(tex.gid)/denom, compression),1.0f);
  
  write_image(d1DLut, input_color, tex.gid);
  
}

DHCR_KERNEL void kernel_make2DLut(
        texture2d_write_t         d2DLut DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t  compression DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG uint_ref_t  clevel DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex;
  get_kernel_texel2d(d2DLut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float qsize = (float)(clevel*clevel);
  float denom = qsize-1;
  
  uint  bindex =
          (floorf((float)(tex.gid.x) / denom)
           + (float)(clevel) * floorf( (float)(tex.gid.y)/denom));
  float b = (float)bindex/denom;
  
  float xindex = floorf((float)(tex.gid.x) / qsize);
  float yindex = floorf((float)(tex.gid.y) / qsize);
  float r = ((float)(tex.gid.x)-xindex*(qsize))/denom;
  float g = ((float)(tex.gid.y)-yindex*(qsize))/denom;
  
  float3 rgb = compress(make_float3(r,g,b),compression);
  
  write_image(d2DLut, to_float4(rgb,1.0f), tex.gid);
  
}

DHCR_KERNEL void kernel_make3DLut(
        texture3d_write_t         d3DLut DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG float2_ref_t  compression DHCR_BIND_BUFFER(1)
        DHCR_KERNEL_GID_3D
) {
  Texel3d tex;
  get_kernel_texel3d(d3DLut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 denom = make_float3(get_texture_width(d3DLut)-1,
                             get_texture_height(d3DLut)-1,
                             get_texture_depth(d3DLut)-1);
  
//  float3 denom = make_float3(get_texture_width(d3DLut),
//                             get_texture_height(d3DLut),
//                             get_texture_depth(d3DLut));

  float4 input_color  = to_float4(compress(to_float3(tex.gid)/denom, compression),1.0f);
  
  write_image(d3DLut, input_color, tex.gid);
  
}

//
// MARK - Resample
//

DHCR_KERNEL void kernel_resample1DLut_to_1DLut(
        texture1d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture1d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture1d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_1D
) {
  Texel1d tex;
  get_kernel_texel1d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 rgb  =  to_float3(read_image(DLutIdentity, tex.gid));
  float x = read_image(DLut,rgb.x).x;
  float y = read_image(DLut,rgb.y).y;
  float z = read_image(DLut,rgb.z).z;
  
  write_image(DLutOut, make_float4(x,y,z,1.0f), tex.gid);
}

/**
    Look up color in Square-like 2D representation of 3D LUT

    @param rgb input color
    @param d2DLut d2DLut Hald-like texture: cube-size*cube-size * level = r-g * b boxed regiones
    @return maped color
    */
inline  DHCR_DEVICE_FUNC float3 sample2DLut(float3 rgb, texture2d_read_t d2DLut){
  
  float  size    = (float)(get_texture_width(d2DLut));
  float  clevel  = roundf(powf((float)size,1.0f/3.0f));
  
  float cube_size = clevel*clevel;
  
  float blueColor = rgb.z * (cube_size-1.0f);//-0.8f);
  
  float2 quad1;
  quad1.y = floorf(floorf(blueColor) / clevel);
  quad1.x = floorf(blueColor) - (quad1.y * clevel);
  
  float2 quad2;
  quad2.y = floorf(ceilf(blueColor) / clevel);
  quad2.x = ceilf(blueColor) - (quad2.y * clevel);
  
  float2 texPos1;
  
  float denom = 1.0f/clevel;
  
  texPos1.x = (quad1.x * denom) + 0.5f/size + ((denom - 1.0f/size) * rgb.x);
  texPos1.y = (quad1.y * denom) + 0.5f/size + ((denom - 1.0f/size) * rgb.y);
  
  float2 texPos2;
  texPos2.x = (quad2.x * denom) + 0.5f/size + ((denom - 1.0f/size) * rgb.x);
  texPos2.y = (quad2.y * denom) + 0.5f/size + ((denom - 1.0f/size) * rgb.y);
  
  float4 newColor1 =  d2DLut.sample(linear_normalized_sampler, texPos1);//d2DLut.sampleread_image(d2DLut, texPos1);
  
  float4 newColor2 =  d2DLut.sample(linear_normalized_sampler, texPos2);//read_image(d2DLut, texPos2);
  
  return to_float3(mix(newColor1, newColor2, fracf(blueColor)));
}


inline  DHCR_DEVICE_FUNC float3 sample2DLut_2v(float3 rgb, texture2d_read_t d2DLut){
  
  float3 textureColor = rgb;
  
  float blueColor = textureColor.b * 63.0f;
  
  float2 quad1;
  quad1.y = floor(floor(blueColor) / 8.0);
  quad1.x = floor(blueColor) - (quad1.y * 8.0);
  
  float2 quad2;
  quad2.y = floor(ceil(blueColor) / 8.0);
  quad2.x = ceil(blueColor) - (quad2.y * 8.0);
  
  float2 texPos1;
  texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.r);
  texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.g);
  
  float2 texPos2;
  texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.r);
  texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.g);
  
  //float4 newColor1 = read_image(d2DLut, texPos1);
  //float4 newColor2 = read_image(d2DLut, texPos2);
  
  constexpr sampler quadSampler3;
  float4 newColor1 = d2DLut.sample(quadSampler3, texPos1);
  constexpr sampler quadSampler4;
  float4 newColor2 = d2DLut.sample(quadSampler4, texPos2);
  
  float4 newColor = mix(newColor1, newColor2, fracf(blueColor));
  return to_float3(newColor);
}


DHCR_KERNEL void kernel_resample2DLut_to_2DLut(
        texture2d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture2d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture2d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex;
  get_kernel_texel2d(DLutOut, tex);
  //if (!get_texel_boundary(tex)) return;
  
  constexpr sampler quadSampler;
  float4 base = DLutIdentity.sample(quadSampler, (float2)tex.gid/((float2)(tex.size)-1.0f));
  
  //float3 rgb  =  to_float3(base);//read_image(DLutIdentity, tex.gid));
  float3 rgb  =  to_float3(read_image(DLut, tex.gid));
  float3 result = rgb;//sample2DLut_2v(rgb, DLut);
  
  write_image(DLutOut, to_float4(result,1.0f), tex.gid);
}

DHCR_KERNEL void kernel_resample3DLut_to_3DLut(
        texture3d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture3d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture3d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_3D
) {
  Texel3d tex;
  get_kernel_texel3d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 rgb  =  to_float3(read_image(DLutIdentity, tex.gid));
  float4 result = read_image(DLut, rgb);
  result.w = 1.0f;
  write_image(DLutOut, result, tex.gid);
}

//
// MARK - Convert
//

// 1D
DHCR_KERNEL void kernel_convert1DLut_to_2DLut(
        texture2d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture2d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture1d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex;
  get_kernel_texel2d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float4 rgba  =  read_image(DLutIdentity, tex.gid);
  float x = read_image(DLut,rgba.x).x;
  float y = read_image(DLut,rgba.y).y;
  float z = read_image(DLut,rgba.z).z;
  
  write_image(DLutOut, make_float4(x,y,z,1.0f), tex.gid);
}

DHCR_KERNEL void kernel_convert1DLut_to_3DLut(
        texture3d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture3d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture1d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_3D
) {
  Texel3d tex;
  get_kernel_texel3d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float4 rgba  =  read_image(DLutIdentity, tex.gid);
  float x = read_image(DLut,rgba.x).x;
  float y = read_image(DLut,rgba.y).y;
  float z = read_image(DLut,rgba.z).z;
  
  write_image(DLutOut, make_float4(x,y,z,1.0f), tex.gid);
}

// 2D
DHCR_KERNEL void kernel_convert2DLut_to_1DLut(
        texture1d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture1d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture2d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_1D
) {
  Texel1d tex;
  get_kernel_texel1d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 rgb  =  to_float3(read_image(DLutIdentity, tex.gid));
  float3 result = sample2DLut(rgb, DLut);
  
  write_image(DLutOut, to_float4(result,1.0f), tex.gid);
}

DHCR_KERNEL void kernel_convert2DLut_to_3DLut(
        texture3d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture3d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture2d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_3D
) {
  Texel3d tex;
  get_kernel_texel3d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 rgb  =  to_float3(read_image(DLutIdentity, tex.gid));
  float3 result = sample2DLut(rgb, DLut);
  
  write_image(DLutOut, to_float4(result,1.0f), tex.gid);
}

//3D
DHCR_KERNEL void kernel_convert3DLut_to_1DLut(
        texture1d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture1d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture3d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_1D
) {
  Texel1d tex;
  get_kernel_texel1d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 rgb  =  to_float3(read_image(DLutIdentity, tex.gid));
  float4 result = read_image(DLut, rgb);
  result.w = 1.0f;
  write_image(DLutOut, result, tex.gid);
}

DHCR_KERNEL void kernel_convert3DLut_to_2DLut(
        texture2d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture2d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture3d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex; get_kernel_texel2d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  float3 rgb    = to_float3(read_image(DLutIdentity, tex.gid));
  
  float4 result = read_image(DLut, rgb);

  result.w = 1.0f;
  
  write_image(DLutOut, result, tex.gid);
}

DHCR_KERNEL void kernel_copy_3DLut(
        texture3d_read_t             d3DLut DHCR_BIND_TEXTURE(0),
        DHCR_DEVICE_ARG float*      buffer DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG uint_ref_t channels DHCR_BIND_BUFFER(2)
        
        DHCR_KERNEL_GID_3D
)
{
  
  Texel3d tex; get_kernel_texel3d(d3DLut, tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4 result = read_image(d3DLut, tex.gid);
  
  uint lut_size = get_texture_width(d3DLut);
  uint index = (tex.gid.x + lut_size * tex.gid.y + lut_size * lut_size * tex.gid.z) * channels;
  buffer[index + 0] = result.x;
  buffer[index + 1] = result.y;
  buffer[index + 2] = result.z;
  buffer[index + 3] = 1.0f;
}

#endif //DEHANCER_GPULIB_CLUT_KERNELS_NEW_H
