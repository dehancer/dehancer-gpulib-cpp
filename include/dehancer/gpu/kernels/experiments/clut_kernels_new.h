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
  
  float3 denom = make_float3(get_texture_width(d1DLut)-1);
  float4 input_color  = make_float4(compress(make_float3(tex.gid)/denom, compression),1);
  
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
  float denom = qsize;
  
  uint  bindex =
          (floorf((float)(tex.gid.x) / denom)
           + (float)(clevel) * floorf( (float)(tex.gid.y)/denom));
  
  float b = (float)bindex/(denom);
  
  float xindex = floorf((float)(tex.gid.x) / (qsize));
  float yindex = floorf((float)(tex.gid.y) / qsize);
  float r = ((float)(tex.gid.x)-xindex*(qsize))/denom;
  float g = ((float)(tex.gid.y)-yindex*(qsize))/denom;
  
  float3 rgb = compress(make_float3(r,g,b),compression);
  
  write_image(d2DLut, make_float4(rgb,1), tex.gid);
  
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

  float4 input_color  = make_float4(compress(make_float3(tex.gid)/denom, compression),1);
  
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
  
  float3 rgb  =  make_float3(read_image(DLutIdentity, tex.gid));
  float x = read_image(DLut,rgb.x).x;
  float y = read_image(DLut,rgb.y).y;
  float z = read_image(DLut,rgb.z).z;
  
  write_image(DLutOut, make_float4(x,y,z,1), tex.gid);
}

/**
    Look up color in Square-like 2D representation of 3D LUT

    @param rgb input color
    @param d2DLut d2DLut Hald-like texture: cube-size*cube-size * level = r-g * b boxed regiones
    @return maped color
    */
inline  DHCR_DEVICE_FUNC float3 sample2DLut(float3 rgb, texture2d_read_t d2DLut){
  
  float  size    = (float)(get_texture_width(d2DLut));
  float  clevel  = (float)(int)roundf(powf((float)size,1.0f/3.0f));

  float cube_size = clevel*clevel;

  float blueColor = rgb.z * (cube_size-1.0f-0.8f);

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

  //constexpr sampler quadSampler1;
  float4 newColor1 = read_image(d2DLut, texPos1);

  //constexpr sampler quadSampler2;
  float4 newColor2 =  read_image(d2DLut, texPos2);

  return make_float3(mix(newColor1, newColor2, fracf(blueColor)));

  
  //  float3 base = rgb;//inputTexture.sample(quadSampler, fragmentInput.textureCoordinate);
//
//  float mul       = clevel*clevel - 1.0f;
//  float blueColor = base.b * mul;
//
//  half2 quad1;
//  quad1.y = floorf(floor(blueColor) / clevel);
//  quad1.x = floorf(blueColor) - (quad1.y * clevel);
//
//  half2 quad2;
//  quad2.y = floorf(ceil(blueColor) / clevel);
//  quad2.x = ceilf(blueColor) - (quad2.y * clevel);
//
//  float2 texPos1;
//  float ret = 0.5f/size;
//  float ret2 = 1.0f/size;
//  float denom = 1.0f/clevel;
//  texPos1.x = (quad1.x * denom) + ret + ((denom - ret2) * base.x);
//  texPos1.y = (quad1.y * denom) + ret + ((denom - ret2) * base.y);
//
//  float2 texPos2;
//  texPos2.x = (quad2.x * denom) + ret + ((denom - ret2) * base.x);
//  texPos2.y = (quad2.y * denom) + ret + ((denom - ret2) * base.y);
//
//  //constexpr sampler quadSampler3;
//  float4 newColor1 = read_image(d2DLut, texPos1); //inputTexture2.sample(quadSampler3, texPos1);
//  //constexpr sampler quadSampler4;
//  float4 newColor2 = read_image(d2DLut, texPos2); //inputTexture2.sample(quadSampler4, texPos2);
//
//  //float4 newColor = mix(newColor1, newColor2, fracf(blueColor));
//  //return mix(base, make_float4(make_float3(newColor), base.w));
//  return make_float3(mix(newColor1, newColor2, fracf(blueColor)));
}

DHCR_KERNEL void kernel_resample2DLut_to_2DLut(
        texture2d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture2d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture2d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_2D
) {
  Texel2d tex;
  get_kernel_texel2d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  //float2 rgb_coords = get_texel_coords(tex);
  //float3 rgb        = make_float3(read_image(DLutIdentity, rgb_coords));
  float3 rgb    = make_float3(sampled_color(DLutIdentity, tex.size, tex.gid));
  float3 result = sample2DLut(rgb, DLut);
  
  write_image(DLutOut, make_float4(result,1.0f), tex.gid);
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
  
  float3 rgb_coords = get_texel_coords(tex);
  
  float3 rgb    = make_float3(read_image(DLutIdentity, rgb_coords));
  
  float4 result = read_image(DLut, rgb);
  result.w = 1.0f;
  write_image(DLutOut, result, tex.gid);
}

DHCR_KERNEL void kernel_resample_hald_3DLut_to_3DLut(
        texture3d_read_t         DLutIdentity DHCR_BIND_TEXTURE(0),
        texture3d_write_t        DLutOut DHCR_BIND_TEXTURE(1),
        texture3d_read_t         DLut DHCR_BIND_TEXTURE(2)
        DHCR_KERNEL_GID_3D
) {
  Texel3d tex;
  get_kernel_texel3d(DLutOut, tex);
  if (!get_texel_boundary(tex)) return;
  
  /***
   * TODO:
   * must be parameterized
   */
  float  l = (1.0f-64.0f/65.0f);
  float3 al = make_float3(l);
  float3 ah = make_float3(1.0f/(1.0f + l*2.0f));
  
  float3 rgb_coords = get_texel_coords(tex);
  rgb_coords = clamp(ah*rgb_coords+al, 0.0f, 1.0f);
  
  float3 rgb    = make_float3(read_image(DLutIdentity, rgb_coords));
  
  //rgb = clamp(ah*rgb+al, 0.0f, 1.0f);
  
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
  
  write_image(DLutOut, make_float4(x,y,z,1), tex.gid);
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
  
  //float4 rgba  =  read_image(DLutIdentity, tex.gid);
  float3 rgb_coords = get_texel_coords(tex);
  float4 rgba    = read_image(DLutIdentity, rgb_coords);
  float x = read_image(DLut,rgba.x).x;
  float y = read_image(DLut,rgba.y).y;
  float z = read_image(DLut,rgba.z).z;
  
  write_image(DLutOut, make_float4(x,y,z,1), tex.gid);
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
  
  float3 rgb    = make_float3(read_image(DLutIdentity, tex.gid));
  float3 result = sample2DLut(rgb, DLut);
  
  write_image(DLutOut, make_float4(result,1.0f), tex.gid);
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
  
  //float3 rgb    = make_float3(read_image(DLutIdentity, tex.gid));
  float3 rgb_coords = get_texel_coords(tex);
  float3 rgb    = make_float3(read_image(DLutIdentity, rgb_coords));
  float3 result = sample2DLut(rgb, DLut);
  
  write_image(DLutOut, make_float4(result,1.0f), tex.gid);
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
  
  float3 rgb  =  make_float3(read_image(DLutIdentity, tex.gid));
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
  
  float3 rgb    = make_float3(sampled_color(DLutIdentity, tex.size, tex.gid));
  
  float4 result = read_image(DLut, rgb);

  result.w = 1.0f;
  
  write_image(DLutOut, result, tex.gid);
}

DHCR_KERNEL void kernel_copy_3DLut(
        texture3d_read_t             d3DLut DHCR_BIND_TEXTURE(0),
        DHCR_DEVICE_ARG float*       buffer DHCR_BIND_BUFFER(1),
        //DHCR_DEVICE_ARG uint_ref_t lut_size DHCR_BIND_BUFFER(2),
        DHCR_DEVICE_ARG uint_ref_t channels DHCR_BIND_BUFFER(2)
        
        DHCR_KERNEL_GID_3D
)
{
  
  Texel3d tex; get_kernel_texel3d(d3DLut, tex);
  
  if (!get_texel_boundary(tex)) return;
  
  float4 result = read_image(d3DLut, tex.gid);
  
  uint len   = get_texture_width(d3DLut);
  uint index = (tex.gid.x + len * tex.gid.y + len * len * tex.gid.z) * channels;
  
  buffer[index + 0] = result.x;
  buffer[index + 1] = result.y;
  buffer[index + 2] = result.z;
  buffer[index + 3] = 1.0f;
}

#endif //DEHANCER_GPULIB_CLUT_KERNELS_NEW_H
