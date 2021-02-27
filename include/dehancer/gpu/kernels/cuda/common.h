//
// Created by denn on 29.12.2020.
//

#pragma once

#include <cmath>
#include "dehancer/gpu/kernels/cuda/texture1d.h"
#include "dehancer/gpu/kernels/cuda/texture2d.h"
#include "dehancer/gpu/kernels/cuda/texture3d.h"
#include "dehancer/gpu/kernels/cuda/cmatrix.h"

#include "dehancer/gpu/kernels/constants.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/type_cast.h"
#include "dehancer/gpu/kernels/hash_utils.h"
#include "dehancer/gpu/kernels/cmath.h"

#define texture1d_read_t DHCR_READ_ONLY image1d_t
#define texture1d_write_t DHCR_WRITE_ONLY image1d_t

#define texture2d_read_t DHCR_READ_ONLY image2d_t
#define texture2d_write_t DHCR_WRITE_ONLY image2d_t

#define texture3d_read_t DHCR_READ_ONLY image3d_t
#define texture3d_write_t DHCR_WRITE_ONLY image3d_t

 inline __device__ int __attribute__((overloadable)) get_texture_width(image1d_t source) {
  return (int)source.get_width();
}

 inline __device__ int __attribute__((overloadable)) get_texture_height(image1d_t source) {
  return (int)1;
}

 inline __device__ int __attribute__((overloadable)) get_texture_depth(image1d_t source) {
  return (int)1;
}

 inline __device__ int __attribute__((overloadable)) get_texture_width(image2d_t source) {
  return (int)source.get_width();
}

 inline __device__ int __attribute__((overloadable)) get_texture_height(image2d_t source) {
  return (int)source.get_height();
}

 inline __device__ int __attribute__((overloadable)) get_texture_depth(image2d_t source) {
  return (int)1;
}

 inline __device__ int __attribute__((overloadable)) get_texture_width(image3d_t source) {
  return (int)source.get_width();
}

 inline __device__ int __attribute__((overloadable)) get_texture_height(image3d_t source) {
  return (int)source.get_height();
}

 inline __device__ int __attribute__((overloadable)) get_texture_depth(image3d_t source) {
  return (int)source.get_depth();
}

inline __device__ void get_kernel_tid1d(int& tid) {
  tid = blockIdx.x * blockDim.x + threadIdx.x;
}

inline __device__  void get_kernel_tid2d(int2& tid) {
  tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tid.y = blockIdx.y * blockDim.y + threadIdx.y;
}

inline __device__  void get_kernel_tid3d(int3& tid) {
  tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tid.y = blockIdx.y * blockDim.y + threadIdx.y;
  tid.z = blockIdx.z * blockDim.z + threadIdx.z;
}

inline __device__  void get_kernel_texel1d(__read_only image1d_t destination, Texel1d& tex) {
  tex.gid = blockIdx.x * blockDim.x + threadIdx.x;
  tex.size = destination.get_width();
}

inline __device__  void get_kernel_texel2d( image2d_t destination, Texel2d& tex) {
  
  tex.gid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tex.gid.y = blockIdx.y * blockDim.y + threadIdx.y;
  
  tex.size.x = destination.get_width();
  tex.size.y = destination.get_height();
}

inline __device__  void get_kernel_texel3d(image3d_t destination, Texel3d& tex) {
  
  tex.gid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tex.gid.y = blockIdx.y * blockDim.y + threadIdx.y;
  tex.gid.z = blockIdx.z * blockDim.z + threadIdx.z;
  
  tex.size.x = destination.get_width();
  tex.size.y = destination.get_height();
  tex.size.z = destination.get_depth();
}

inline __device__   bool get_texel_boundary(Texel1d tex) {
  return tex.gid < tex.size;
}

inline __device__   bool get_texel_boundary(Texel2d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y) {
    return false;
  }
  return true;
}

inline __device__   bool get_texel_boundary(Texel3d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y || tex.gid.z >= tex.size.z) {
    return false;
  }
  return true;
}

inline __device__   float get_texel_coords(Texel1d tex) {
  return (float)tex.gid / (float)(tex.size - 1);
}

inline __device__   float2 get_texel_coords(Texel2d tex) {
  return (float2){(float)tex.gid.x / (float)(tex.size.x - 1),
                  (float)tex.gid.y / (float)(tex.size.y - 1)};
}

inline __device__   float3 get_texel_coords(Texel3d tex) {
  return (float3){
          (float)tex.gid.x / (float)(tex.size.x - 1),
          (float)tex.gid.y / (float)(tex.size.y - 1),
          (float)tex.gid.z / (float)(tex.size.z - 1)
  };
}

template<class T>
__device__ const T& clamp(const T& v, const T& lo, const T& hi )
{
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

// 1D
inline __device__ float4 __attribute__((overloadable)) read_image( const image1d_t& source, int gid) {
  return source.read_pixel(gid);
}

inline __device__ float4 __attribute__((overloadable)) read_image( const image1d_t& source, float coords) {
  return source.read(coords);
}

inline __device__ float4 __attribute__((overloadable)) read_image( const image1d_t& source, float4 coords) {
  float4 color = make_float4(1,1,1,1);
  color.x = source.read(coords.x).x;
  color.y = source.read(coords.y).x;
  color.z = source.read(coords.z).x;
  color.w = 1.0f;
  return color;
}

inline __device__ void __attribute__((overloadable)) write_image( image1d_t& destination, float4 color, int gid) {
  destination.write(color, gid);
}

// 2D
inline __device__  float4 __attribute__((overloadable)) read_image( const image2d_t& source, int2 gid) {
  return source.read_pixel(gid);
}

inline __device__ float4 __attribute__((overloadable)) read_image( const image2d_t& source, float2 coords) {
  return source.read(coords);
}

inline __device__  void __attribute__((overloadable)) write_image(  image2d_t& destination, float4 color, int2 gid) {
  destination.write(color, gid);
}

// 3D
inline __device__  float4 __attribute__((overloadable)) read_image( const image3d_t& source, int3 gid) {
  return source.read_pixel(gid);
}

inline __device__  float4 __attribute__((overloadable)) read_image( const image3d_t& source, float3 coords) {
  return source.read(coords);
}

inline __device__  float4 __attribute__((overloadable)) read_image( const image3d_t& source, float4 coords) {
  return source.read((float3){coords.x,coords.y,coords.z});
}

inline __device__  void __attribute__((overloadable)) write_image(  image3d_t& destination, float4 color, int3 gid) {
  destination.write(color, gid);
}

inline __device__   float3 compress(float3 rgb, float2 compression) {
  return  compression.x*rgb + compression.y;
}

