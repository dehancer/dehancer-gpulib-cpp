//
// Created by denn nevera on 30/11/2020.
//

#ifndef DEHANCER_GPULIB_OPENCL_COMMON_H
#define DEHANCER_GPULIB_OPENCL_COMMON_H

#include "dehancer/gpu/kernels/constants.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/type_cast.h"
#include "dehancer/gpu/kernels/hash_utils.h"
#include "dehancer/gpu/kernels/cmath.h"

constexpr sampler linear_normalized_sampler(address::mirrored_repeat, filter::linear, coord::normalized);
constexpr sampler nearest_sampler(address::clamp_to_border, filter::nearest, coord::pixel);

#define texture1d_read_t  texture1d<float, access::sample>
#define texture1d_write_t texture1d<float, access::write>

#define texture2d_read_t  texture2d<float, access::sample>
#define texture2d_write_t texture2d<float, access::write>

#define texture3d_read_t  texture3d<float, access::sample>
#define texture3d_write_t texture3d<float, access::write>

#define  get_kernel_tid1d(tid) { \
  tid = int(__dehancer_kernel_gid_1d__);\
}

#define  get_kernel_tid2d(tid) { \
  tid = int2(__dehancer_kernel_gid_2d__.x, __dehancer_kernel_gid_2d__.y);\
}

#define  get_kernel_tid3d(tid) { \
  tid = int3(__dehancer_kernel_gid_3d__.x, __dehancer_kernel_gid_3d__.y, __dehancer_kernel_gid_3d__.z);  \
}

#define get_kernel_texel1d(destination, tex) { \
  tex.gid =  int(__dehancer_kernel_gid_1d__); \
  tex.size = int(destination.get_width()); \
}

#define get_kernel_texel2d(destination, tex) { \
  get_kernel_tid2d(tex.gid);\
  tex.size = int2(destination.get_width(), destination.get_height()); \
}

#define get_kernel_texel3d(destination, tex) { \
  get_kernel_tid3d(tex.gid);\
  tex.size = int3(destination.get_width(), destination.get_height(), destination.get_depth()); \
}

#define get_texture_width(image)  image.get_width()
#define get_texture_height(image) image.get_height()
#define get_texture_depth(image)  image.get_depth()

static inline  bool __attribute__((overloadable)) get_texel_boundary(Texel1d tex) {
  if (tex.gid >= tex.size) {
    return false;
  }
  return true;
}

static inline  bool __attribute__((overloadable)) get_texel_boundary(Texel2d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y) {
    return false;
  }
  return true;
}

static inline  bool __attribute__((overloadable)) get_texel_boundary(Texel3d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y || tex.gid.z >= tex.size.z) {
    return false;
  }
  return true;
}

static inline  float __attribute__((overloadable)) get_texel_coords(Texel1d tex) {
  return float(tex.gid) / float(tex.size - 1);
}

static inline  float2 __attribute__((overloadable)) get_texel_coords(Texel2d tex) {
  return (float2){(float)tex.gid.x / (float)(tex.size.x - 1),
                  (float)tex.gid.y / (float)(tex.size.y - 1)};
}

static inline  float3 __attribute__((overloadable)) get_texel_coords(Texel3d tex) {
  return (float3){
          (float)tex.gid.x / (float)(tex.size.x - 1),
          (float)tex.gid.y / (float)(tex.size.y - 1),
          (float)tex.gid.z / (float)(tex.size.z - 1),
  };
}

// 1D
static inline float4 __attribute__((overloadable)) read_image(texture1d_read_t source, int gid) {
  return source.sample(nearest_sampler, gid);
}

static inline float4 __attribute__((overloadable)) read_image(texture1d_read_t source, float coords) {
  return source.sample(linear_normalized_sampler, coords);
}

static inline float4 __attribute__((overloadable)) read_image(texture1d_read_t source, float4 coords) {
  float4 color = coords;
  color.x = source.sample(linear_normalized_sampler, color.x).x;
  color.y = source.sample(linear_normalized_sampler, color.y).y;
  color.z = source.sample(linear_normalized_sampler, color.z).z;
  return color;
}

static inline void __attribute__((overloadable)) write_image(texture1d_write_t destination, float4 color, int gid) {
  destination.write(color, (uint)gid);
}


// 2D
static inline float4 __attribute__((overloadable)) read_image(texture2d_read_t source, int2 gid) {
  return source.sample(nearest_sampler, (float2)gid);
}

static inline float4 __attribute__((overloadable)) read_image(texture2d_read_t source, float2 coords) {
  return source.sample(linear_normalized_sampler, coords);
}

static inline void __attribute__((overloadable)) write_image(texture2d_write_t destination, float4 color, int2 gid) {
  destination.write(color, (uint2)gid);
}

// 3D
static inline float4 __attribute__((overloadable)) read_image(texture3d_read_t source, int3 gid) {
  return source.sample(nearest_sampler, (float3)gid);
}

static inline float4 __attribute__((overloadable)) read_image(texture3d_read_t source, float3 coords) {
  return source.sample(linear_normalized_sampler, coords);
}

static inline float4 __attribute__((overloadable)) read_image(texture3d_read_t source, float4 coords) {
  return source.sample(linear_normalized_sampler, coords.xyz);
}

static inline void __attribute__((overloadable)) write_image(texture3d_write_t destination, float4 color, int3 gid) {
  destination.write(color, (uint3)gid);
}


static inline  float3 compress(float3 rgb, float2 compression) {
  return  compression.x*rgb + compression.y;
}


#endif //DEHANCER_GPULIB_OPENCL_COMMON_H