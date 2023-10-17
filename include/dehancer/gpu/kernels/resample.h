//
// Created by denn on 20.01.2021.
//

#ifndef DEHANCER_GPULIB_RESAMPLE_H
#define DEHANCER_GPULIB_RESAMPLE_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

#define DHCR_AXIS_OFFSET (-0.5f)

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

//  float  u = ceilf(x);
//  float  v = ceilf(y);

  float  px = x-u;
  float  py = y-v;
  
  int dx = 1;
  int dy = 1;
  
  int2   gid_src = make_int2(u,v);
  
  float4 q11 = read_image(source, gid_src+make_int2(0, 0 ));
  float4 q21 = read_image(source, gid_src+make_int2(dx,0 ));

  float4 q12 = read_image(source, gid_src+make_int2(0, dy));
  float4 q22 = read_image(source, gid_src+make_int2(dx,dy));

  float4 c1  = mix(q11,q21,px);
  float4 c2  = mix(q12,q22,px);

  return mix(c1,c2,py);
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
  
  float  u = floorf(x);
  float  v = floorf(y);
  float  w = floorf(z);
  
  float  px = x - u;
  float  py = y - v;
  float  pz = z - w;
  
  int dx = 1;
  int dy = 1;
  int dw = 1;
  
  int3   gid_src = make_int3(u,v,w);
  
  float4 q111 = read_image(source, gid_src+make_int3(0, 0,  0));
  float4 q121 = read_image(source, gid_src+make_int3(0, dy, 0));
  float4 q221 = read_image(source, gid_src+make_int3(dx,dy, 0));
  float4 q211 = read_image(source, gid_src+make_int3(dx,0,  0));
  
  float4 q112 = read_image(source, gid_src+make_int3(0, 0,  dw));
  float4 q122 = read_image(source, gid_src+make_int3(0, dy, dw));
  float4 q222 = read_image(source, gid_src+make_int3(dx,dy, dw));
  float4 q212 = read_image(source, gid_src+make_int3(dx,0,  dw));
  
  
  float4 c1  = mix(q111,q211,to_float4(px));
  float4 c2  = mix(q121,q221,to_float4(px));
  
  float4 c3  = mix(q112,q212,to_float4(pz));
  float4 c4  = mix(q122,q222,to_float4(pz));
  
  float4 c5 = mix(c1,c2,to_float4(py));
  
  return mix(c3,c4,c5);
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
float4 tex2D_smooth_bicubic(texture2d_read_t tex, float x, float y)
{
  x += DHCR_AXIS_OFFSET;
  y += DHCR_AXIS_OFFSET;

  float px = floorf(x);
  float py = floorf(y);
  float fx = x - px;
  float fy = y - py;

  float2 gid = make_float2(px,py);
  float2 fx2 = make_float2(get_texture_width(tex)-1, get_texture_height(tex)-1);

  float4 c0 = cubicFilter(fx,
                          read_image(tex, (gid+make_float2(-1,-1)) /fx2 ),
                          read_image(tex, (gid+make_float2( 0,-1)) /fx2 ),
                          read_image(tex, (gid+make_float2(+1,-1)) /fx2 ),
                          read_image(tex, (gid+make_float2(+2,-1)) /fx2 ));
  float4 c1 = cubicFilter(fx,
                          read_image(tex, (gid+make_float2(-1, 0)) /fx2 ),
                          read_image(tex, (gid+make_float2( 0, 0)) /fx2 ),
                          read_image(tex, (gid+make_float2(+1, 0)) /fx2 ),
                          read_image(tex, (gid+make_float2(+2, 0)) /fx2 ));
  float4 c2 =  cubicFilter(fx,
                           read_image(tex, (gid+make_float2(-1,+1)) /fx2 ),
                           read_image(tex, (gid+make_float2( 0,+1)) /fx2 ),
                           read_image(tex, (gid+make_float2(+1,+1)) /fx2 ),
                           read_image(tex, (gid+make_float2(+2,+1)) /fx2 ));
  float4 c3 =  cubicFilter(fx,
                           read_image(tex, (gid+make_float2(-1,+2)) /fx2 ),
                           read_image(tex, (gid+make_float2( 0,+2)) /fx2 ),
                           read_image(tex, (gid+make_float2(+1,+2)) /fx2 ),
                           read_image(tex, (gid+make_float2(+2,+2)) /fx2 ));

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
