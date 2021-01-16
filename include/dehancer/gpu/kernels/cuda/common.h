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

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_width(const image1d_t& source) {
  return (int)source.get_width();
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_height(const image1d_t& source) {
  return (int)1;
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_depth(const image1d_t& source) {
  return (int)1;
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_width(const image2d_t& source) {
  return (int)source.get_width();
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_height(const image2d_t& source) {
  return (int)source.get_height();
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_depth(const image2d_t& source) {
  return (int)1;
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_width(const image3d_t& source) {
  return (int)source.get_width();
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_height(const image3d_t& source) {
  return (int)source.get_height();
}

static inline __device__ __host__ int __attribute__((overloadable)) get_texture_depth(const image3d_t& source) {
  return (int)source.get_depth();
}

inline __device__ __host__ void get_kernel_tid1d(int& tid) {
  tid = blockIdx.x * blockDim.x + threadIdx.x;
}

inline __device__ __host__ void get_kernel_tid2d(int2& tid) {
  tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tid.y = blockIdx.y * blockDim.y + threadIdx.y;
}

inline __device__ __host__ void get_kernel_tid3d(int3& tid) {
  tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tid.y = blockIdx.y * blockDim.y + threadIdx.y;
  tid.z = blockIdx.z * blockDim.z + threadIdx.z;
}

inline __device__ __host__ void get_kernel_texel1d(__read_only image1d_t destination, Texel1d& tex) {
  tex.gid = blockIdx.x * blockDim.x + threadIdx.x;
  tex.size = destination.get_width();
}

inline __device__ __host__ void get_kernel_texel2d(__read_only image2d_t destination, Texel2d& tex) {

  tex.gid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tex.gid.y = blockIdx.y * blockDim.y + threadIdx.y;

  tex.size.x = destination.get_width();
  tex.size.y = destination.get_height();
}

inline __device__ __host__ void get_kernel_texel3d(dehancer::nvcc::texture3d<float4> destination, Texel3d& tex) {

  tex.gid.x = blockIdx.x * blockDim.x + threadIdx.x;
  tex.gid.y = blockIdx.y * blockDim.y + threadIdx.y;
  tex.gid.z = blockIdx.z * blockDim.z + threadIdx.z;

  tex.size.x = destination.get_width();
  tex.size.y = destination.get_height();
  tex.size.z = destination.get_depth();
}

inline __device__ __host__  bool get_texel_boundary(Texel1d tex) {
  return tex.gid < tex.size;
}

inline __device__ __host__  bool get_texel_boundary(Texel2d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y) {
    return false;
  }
  return true;
}

inline __device__ __host__  bool get_texel_boundary(Texel3d tex) {
  if (tex.gid.x >= tex.size.x || tex.gid.y >= tex.size.y || tex.gid.z >= tex.size.z) {
    return false;
  }
  return true;
}

inline __device__ __host__  float get_texel_coords(Texel1d tex) {
  return (float)tex.gid / (float)(tex.size - 1);
}

inline __device__ __host__  float2 get_texel_coords(Texel2d tex) {
  return (float2){(float)tex.gid.x / (float)(tex.size.x - 1),
                  (float)tex.gid.y / (float)(tex.size.y - 1)};
}

inline __device__ __host__  float3 get_texel_coords(Texel3d tex) {
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
inline __device__ float4 __attribute__((overloadable)) read_image(__read_only image1d_t source, int gid) {
  float size   = (float)source.get_width();
  float coords = (float)gid;
  return source.read(coords/size);
}

inline __device__ float4 __attribute__((overloadable)) read_image(__read_only image1d_t source, float coords) {
  return source.read(coords);
}

inline __device__ float4 __attribute__((overloadable)) read_image(__read_only image1d_t source, float4 coords) {
  float4 color = coords;
  color.x = source.read(color.x).x;
  color.y = source.read(color.y).y;
  color.z = source.read(color.z).z;
  return color;
}

inline __device__ void __attribute__((overloadable)) write_image(__write_only image1d_t destination, float4 color, int gid) {
  destination.write(color, gid);
}


// 2D
inline __device__ __host__ float4 __attribute__((overloadable)) read_image(__read_only image2d_t source, int2 gid) {
  float2 size   = (float2){(float)source.get_width(),(float)source.get_height()};
  float2 coords = (float2){(float)gid.x,(float)gid.y};
  return source.read(coords/size);
}

inline __device__ float4 __attribute__((overloadable)) read_image(__read_only image2d_t source, float2 coords) {
  return source.read(coords);
}

inline __device__ __host__ void __attribute__((overloadable)) write_image(__write_only image2d_t destination, float4 color, int2 gid) {
  destination.write(color, gid);
}

// 3D
inline __device__ __host__ float4 __attribute__((overloadable)) read_image(__read_only image3d_t source, int3 gid) {
  float3 size   = (float3){(float)source.get_width(),(float)source.get_height(),(float)source.get_depth()};
  float3 coords = (float3){(float)gid.x,(float)gid.y,(float)gid.z};
  return source.read(coords/size);
}

inline __device__ __host__ float4 __attribute__((overloadable)) read_image(__read_only image3d_t source, float3 coords) {
  return source.read(coords);
}

inline __device__ __host__ float4 __attribute__((overloadable)) read_image(__read_only image3d_t source, float4 coords) {
  return source.read((float3){coords.x,coords.y,coords.z});
}

inline __device__ __host__ void __attribute__((overloadable)) write_image(__write_only image3d_t destination, float4 color, int3 gid) {
  destination.write(color, gid);
}

inline __device__ __host__ float4 sampled_color(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int2 gid
){

  Texel2d tex; get_kernel_texel2d(destination,tex);

  float2 coords = get_texel_coords(tex);

  return read_image(source, coords);
}

inline __device__ __host__  float3 compress(float3 rgb, float2 compression) {
  return  compression.x*rgb + compression.y;
}

