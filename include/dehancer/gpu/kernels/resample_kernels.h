//
// Created by denn on 19.01.2021.
//

#ifndef DEHANCER_GPULIB_RESAMPLE_KERNELS_H
#define DEHANCER_GPULIB_RESAMPLE_KERNELS_H

#include "dehancer/gpu/kernels/common.h"


DHCR_KERNEL void kernel_bilinear(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1)
){
  Texel2d tex_dest; get_kernel_texel2d(destination,tex_dest);
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  if (!get_texel_boundary(tex_dest)) return;
  
  float2 coords = get_texel_coords(tex_dest);
  
  float2 coords_src = coords * make_float2(tex_src.size);
  
  float  u = floor(coords_src.x);
  float  v = floor(coords_src.y);
  
  float  px = coords_src.x - u;
  float  py = coords_src.y - v;

//  int dx = px<0.5 ? -1: 1;
//  int dy = py<0.5 ? -1: 1;
  
  int dx = 1;
  int dy = 1;
  
  int2   gid_src = make_int2(u,v);
  
  float4 q11 = read_image(source, gid_src+make_int2(0,0));
  float4 q12 = read_image(source, gid_src+make_int2(0,dy));
  float4 q22 = read_image(source, gid_src+make_int2(dx,dy));
  float4 q21 = read_image(source, gid_src+make_int2(dx,0));
  
  float4 c1  = mix(q11,q21,make_float4(px));
  float4 c2  = mix(q12,q22,make_float4(px));
  
  float4 color = mix(c1,c2,make_float4(py));
  write_image(destination, color, tex_dest.gid);
}

//// w0, w1, w2, and w3 are the four cubic B-spline basis functions
inline DHCR_DEVICE_FUNC
float w0(float a)
{
  //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
  return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

inline DHCR_DEVICE_FUNC
float w1(float a)
{
  //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
  return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

inline DHCR_DEVICE_FUNC
float w2(float a)
{
  //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
  return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

inline DHCR_DEVICE_FUNC
float w3(float a)
{
  return (1.0f/6.0f)*(a*a*a);
}

inline DHCR_DEVICE_FUNC
float4 cubicFilter(float x, float4 c0, float4  c1, float4  c2, float4  c3)
{
  float4 r;
  r = c0 * make_float4(w0(x));
  r += c1 * make_float4(w1(x));
  r += c2 * make_float4(w2(x));
  r += c3 * make_float4(w3(x));
  return r;
}

inline DHCR_DEVICE_FUNC
float4 tex2D_bicubic(texture2d_read_t tex, float x, float y)
{
  x -= 0.5f;
  y -= 0.5f;
  float px = floor(x);
  float py = floor(y);
  float fx = x - px;
  float fy = y - py;
  
  int2 gid = make_int2(px,py);
  
  float4 c0 = cubicFilter(fx,
                          read_image(tex, gid+make_int2(-1,-1)),
                          read_image(tex, gid+make_int2( 0,-1)),
                          read_image(tex, gid+make_int2(+1,-1)),
                          read_image(tex, gid+make_int2(+2,-1)));
  float4 c1 = cubicFilter(fx,
                          read_image(tex, gid+make_int2(-1, 0)),
                          read_image(tex, gid+make_int2( 0, 0)),
                          read_image(tex, gid+make_int2(+1, 0)),
                          read_image(tex, gid+make_int2(+2, 0)));
  float4 c2 =  cubicFilter(fx,
                           read_image(tex, gid+make_int2(-1,+1)),
                           read_image(tex, gid+make_int2( 0,+1)),
                           read_image(tex, gid+make_int2(+1,+1)),
                           read_image(tex, gid+make_int2(+2,+1)));
  float4 c3 =  cubicFilter(fx,
                           read_image(tex, gid+make_int2(-1,+2)),
                           read_image(tex, gid+make_int2( 0,+2)),
                           read_image(tex, gid+make_int2(+1,+2)),
                           read_image(tex, gid+make_int2(+2,+2)));
  
  return cubicFilter(fy,c0,c1,c2,c3);
}

DHCR_KERNEL void kernel_bicubic(
        texture2d_read_t       source DHCR_BIND_TEXTURE(0),
        texture2d_write_t destination DHCR_BIND_TEXTURE(1)
){
  Texel2d tex_dest; get_kernel_texel2d(destination,tex_dest);
  Texel2d tex_src;  get_kernel_texel2d(source,tex_src);
  
  if (!get_texel_boundary(tex_dest)) return;
  
  float2 coords = get_texel_coords(tex_dest);
  
  float2 coords_src = coords * make_float2(tex_src.size);
  
  float4 color = tex2D_bicubic(source, coords_src.x, coords_src.y);
  
  write_image(destination, color, tex_dest.gid);
}

#endif //DEHANCER_GPULIB_RESAMPLE_KERNELS_H
