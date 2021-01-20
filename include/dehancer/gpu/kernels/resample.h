//
// Created by denn on 20.01.2021.
//

#ifndef DEHANCER_GPULIB_RESAMPLE_H
#define DEHANCER_GPULIB_RESAMPLE_H

#include "dehancer/gpu/kernels/common.h"

/***
 * Bilinear interpolation
 * @param source - source texture
 * @param x - source x pixel coord
 * @param y - source y pixel coord
 * @return color
 */
inline DHCR_DEVICE_FUNC
float4 tex2D_bilinear(texture2d_read_t source, float x, float y)
{
  x -= 0.5f;
  y -= 0.5f;
  
  float  u = floor(x);
  float  v = floor(y);
  
  float  px = x - u;
  float  py = y - v;
  
  int dx = 1;
  int dy = 1;
  
  int2   gid_src = make_int2(u,v);
  
  float4 q11 = read_image(source, gid_src+make_int2(0,0));
  float4 q12 = read_image(source, gid_src+make_int2(0,dy));
  float4 q22 = read_image(source, gid_src+make_int2(dx,dy));
  float4 q21 = read_image(source, gid_src+make_int2(dx,0));
  
  float4 c1  = mix(q11,q21,make_float4(px));
  float4 c2  = mix(q12,q22,make_float4(px));
  
  return mix(c1,c2,make_float4(py));
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

/***
 * Bicubic interpolation
 * @param source - source texture
 * @param x - source x pixel coord
 * @param y - source y pixel coord
 * @return color
 */
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

inline DHCR_DEVICE_FUNC
float4 tex2D_box_average(texture2d_read_t tex, float x, float y)
{
//  x -= 0.5f;
//  y -= 0.5f;
  float px = floor(x);
  float py = floor(y);

  int2 gid = make_int2(px,py);
  
  float4 xy;
  xy  = read_image(tex, gid + make_int2(-1,-1));
  xy += read_image(tex, gid + make_int2( 0,-1));
  xy += read_image(tex, gid + make_int2( 1,-1));
  xy += read_image(tex, gid + make_int2(-1, 0));
  xy += read_image(tex, gid + make_int2( 0, 0));
  xy += read_image(tex, gid + make_int2( 1, 0));
  xy += read_image(tex, gid + make_int2(-1, 1));
  xy += read_image(tex, gid + make_int2( 0, 1));
  xy += read_image(tex, gid + make_int2( 1, 1));
  
  float3 w = make_float3(1.0f/9.0f);
  
  return make_float4((make_float3(xy) * w),1.0f);
}

#endif //DEHANCER_GPULIB_RESAMPLE_H
