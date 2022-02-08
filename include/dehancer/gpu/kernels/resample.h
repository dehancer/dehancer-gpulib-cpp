//
// Created by denn on 20.01.2021.
//

#ifndef DEHANCER_GPULIB_RESAMPLE_H
#define DEHANCER_GPULIB_RESAMPLE_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

#define DHCR_AXIS_OFFSET 0.0f // -0.5f

/***
 * Bilinear interpolation
 * @param source - source texture
 * @param x - source x pixel coord
 * @param y - source y pixel coord
 * @return color
 */
inline DHCR_DEVICE_FUNC
float4  tex2D_bilinear(texture2d_read_t source, float x, float y)
{
  x += DHCR_AXIS_OFFSET;
  y += DHCR_AXIS_OFFSET;
  
  float  u = floorf(x);
  float  v = floorf(y);
  
  float  px = x - u;
  float  py = y - v;
  
  int dx = 1;
  int dy = 1;
  
  int2   gid_src = make_int2(u,v);
  
  float4 q11 = read_image(source, gid_src+make_int2(0, 0 ));
  float4 q12 = read_image(source, gid_src+make_int2(0, dy));
  float4 q22 = read_image(source, gid_src+make_int2(dx,dy));
  float4 q21 = read_image(source, gid_src+make_int2(dx,0 ));
  
  float4 c1  = mix(q11,q21,to_float4(px));
  float4 c2  = mix(q12,q22,to_float4(px));
  
  return mix(c1,c2,to_float4(py));
}

/***
 * Bilinear interpolation
 * @param source - source texture
 * @param x - source x pixel coord
 * @param y - source y pixel coord
 * @return color
 */
inline DHCR_DEVICE_FUNC
float4 tex3D_trilinear(texture3d_read_t source, float x, float y, float z)
{
  x += DHCR_AXIS_OFFSET;
  y += DHCR_AXIS_OFFSET;
  z += DHCR_AXIS_OFFSET;
  
  float3 dim = make_float3(get_texture_width(source),
                           get_texture_height(source),
                           get_texture_depth(source));
  
  float  u = floorf(x);
  float  v = floorf(y);
  float  w = floorf(z);
  
  //float  px = x - u;
  //float  py = y - v;
  //float  pw = z - w;
  
  int dx = 1;
  int dy = 1;
  int dw = 1;
  
  int3   gid_src = make_int3(u,v,w);
  
  float3 R0G0B0 = to_float3(read_image(source, gid_src+make_int3(0,  0,  0 )));
  float3 R0G0B1 = to_float3(read_image(source, gid_src+make_int3(0,  0,  dw)));
  float3 R1G0B0 = to_float3(read_image(source, gid_src+make_int3(dx, 0,  0 )));
  float3 R0G1B0 = to_float3(read_image(source, gid_src+make_int3(0,  dy, 0 )));
  
  float3 R1G0B1 = to_float3(read_image(source, gid_src+make_int3(dx, 0,  dw)));
  float3 R1G1B0 = to_float3(read_image(source, gid_src+make_int3(dx, dy, 0 )));
  float3 R0G1B1 = to_float3(read_image(source, gid_src+make_int3(0,  dy, dw)));
  
  float3 R1G1B1 = to_float3(read_image(source, gid_src+make_int3(dx, dy, dw)));
  
  float R0 = R0G0B0.x * dim.x;
  float G0 = R0G0B0.y * dim.y;
  float B0 = R0G0B0.z * dim.z;
  
  float R1 = R1G0B0.x * dim.x;
  float G1 = R0G1B0.y * dim.y;
  float B1 = R0G0B1.z * dim.z;
  
  float tr = (x-R0)/(R1-R0);
  float tg = (y-G0)/(G1-G0);
  float tb = (z-B0)/(B1-B0);
  
  float3 c0 = R0G0B0;
  float3 c1 = R0G0B1 - R0G0B0;
  float3 c2 = R1G0B0 - R0G0B0;
  float3 c3 = R0G1B0 - R0G0B0;
  float3 c4 = R1G0B1 - R1G0B0 - R0G0B1 + R0G0B0;
  float3 c5 = R1G1B0 - R0G1B0 - R1G0B0 + R0G0B0;
  float3 c6 = R0G1B1 - R0G1B0 - R0G0B1 + R0G0B0;
  float3 c7 = R1G1B1 - R1G1B0 - R0G1B1 - R1G0B1 + R0G0B1 + R0G1B0 + R1G0B0 - R0G0B0;
  
  float3 rgb = c0 + c1*tb + c2*tr + c3*tg + c4*tb*tr + c5*tr*tg + c6*tg*tb + c7*tr*tg*tb;
  
  return to_float4(rgb,1.0f);
}

inline DHCR_DEVICE_FUNC
float read_channel(DHCR_DEVICE_ARG float* source, int2 size, int2 gid){
  int index = gid.y * size.x + gid.x;
  if (index<0) return 0.0f;
  int l = size.x*size.y;
  if (index>=l) return 0.0f;
  return source[index];
}

inline DHCR_DEVICE_FUNC
float channel_bilinear(DHCR_DEVICE_ARG float* source, int2 size, float x, float y)
{
  x += DHCR_AXIS_OFFSET;
  y += DHCR_AXIS_OFFSET;
  
  float  u = floorf(x);
  float  v = floorf(y);
  
  float  px = x - u;
  float  py = y - v;
  
  int dx = 1;
  int dy = 1;
  
  int2   gid_src = make_int2(u,v);
  
  float q11 = read_channel(source, size, gid_src+make_int2(0,  0));
  float q12 = read_channel(source, size, gid_src+make_int2(0, dy));
  float q22 = read_channel(source, size, gid_src+make_int2(dx,dy));
  float q21 = read_channel(source, size, gid_src+make_int2(dx, 0));
  
  float c1  = mix(q11,q21,px);
  float c2  = mix(q12,q22,px);
  
  return mix(c1,c2,py);
}


//// w0, w1, w2, and w3 are the four cubic B-spline basis functions
inline DHCR_DEVICE_FUNC
float w0(float a)
{
  return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

inline DHCR_DEVICE_FUNC
float w1(float a)
{
  return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

inline DHCR_DEVICE_FUNC
float w2(float a)
{
  return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

inline DHCR_DEVICE_FUNC
float w3(float a)
{
  return (1.0f/6.0f)*(a*a*a);
}

inline DHCR_DEVICE_FUNC
float4 __attribute__((overloadable))  cubicFilter(float x, float4 c0, float4  c1, float4  c2, float4  c3)
{
  float4 r = c0 * to_float4(w0(x));
  r += c1 * to_float4(w1(x));
  r += c2 * to_float4(w2(x));
  r += c3 * to_float4(w3(x));
  return r;
}

inline DHCR_DEVICE_FUNC
float __attribute__((overloadable))  cubicFilter(float x, float c0, float  c1, float  c2, float  c3)
{
  float r = c0 * w0(x);
  r += c1 * w1(x);
  r += c2 * w2(x);
  r += c3 * w3(x);
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
  x += DHCR_AXIS_OFFSET;
  y += DHCR_AXIS_OFFSET;

  float px = floorf(x);
  float py = floorf(y);
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
float channel_bicubic(DHCR_DEVICE_ARG float* source, int2 size, float x, float y)
{
  x += DHCR_AXIS_OFFSET;
  y += DHCR_AXIS_OFFSET;
  
  float px = floorf(x);
  float py = floorf(y);
  float fx = x - px;
  float fy = y - py;
  
  int2 gid = make_int2(px,py);
  
  float c0 = cubicFilter(fx,
                          read_channel(source, size, gid+make_int2(-1,-1)),
                          read_channel(source, size, gid+make_int2( 0,-1)),
                          read_channel(source, size, gid+make_int2(+1,-1)),
                          read_channel(source, size, gid+make_int2(+2,-1)));
  float c1 = cubicFilter(fx,
                          read_channel(source, size, gid+make_int2(-1, 0)),
                          read_channel(source, size, gid+make_int2( 0, 0)),
                          read_channel(source, size, gid+make_int2(+1, 0)),
                          read_channel(source, size, gid+make_int2(+2, 0)));
  float c2 =  cubicFilter(fx,
                           read_channel(source, size, gid+make_int2(-1,+1)),
                           read_channel(source, size, gid+make_int2( 0,+1)),
                           read_channel(source, size, gid+make_int2(+1,+1)),
                           read_channel(source, size, gid+make_int2(+2,+1)));
  float c3 =  cubicFilter(fx,
                           read_channel(source, size, gid+make_int2(-1,+2)),
                           read_channel(source, size, gid+make_int2( 0,+2)),
                           read_channel(source, size, gid+make_int2(+1,+2)),
                           read_channel(source, size, gid+make_int2(+2,+2)));
  
  return cubicFilter(fy,c0,c1,c2,c3);
}


inline DHCR_DEVICE_FUNC
float4 tex2D_box_average(texture2d_read_t tex, float x, float y)
{
  x += DHCR_AXIS_OFFSET;
  y += DHCR_AXIS_OFFSET;
  
  float px = floorf(x);
  float py = floorf(y);

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
  
  float3 w = to_float3(1.0f/9.0f);
  
  return to_float4((to_float3(xy) * w),1.0f);
}

#endif //DEHANCER_GPULIB_RESAMPLE_H
