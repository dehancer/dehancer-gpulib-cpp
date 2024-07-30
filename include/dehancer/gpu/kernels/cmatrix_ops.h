//
// Created by Dennis Svinarchuk on 30.07.24.
//

#ifndef DEHANCER_GPULIB_CMATRIX_OPS_H
#define DEHANCER_GPULIB_CMATRIX_OPS_H


#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

static inline DHCR_DEVICE_FUNC float3 float3_multiply_float3x3( float3 v, float3x3 M)
{
#if defined(__CUDA_ARCH__)
  return M*v;
#elif defined(__METAL_VERSION__)
  return v*M;
#elif defined(CL_VERSION_1_2)
  #if DEHANCER_GPU_CODE
  return make_float3(
          M.m11*v.x + M.m12*v.y + M.m13*v.z,
          M.m21*v.x + M.m22*v.y + M.m23*v.z,
          M.m31*v.x + M.m32*v.y + M.m33*v.z
  );
#else
  return v*M;
#endif
#else
  return v*M;
#endif
}

static inline DHCR_DEVICE_FUNC
float3x3 float3x3_multiply_float3x3( float3x3 M,  float3x3 N)
{
#if defined(__CUDA_ARCH__)
  return M*N;
#elif defined(__METAL_VERSION__)
  return M*N;
#elif defined(CL_VERSION_1_2)
  #if DEHANCER_GPU_CODE
  return (float3x3){
          M.m11*N.m11+M.m12*N.m21+M.m13*N.m31, M.m11*N.m12+M.m12*N.m22+M.m13*N.m32, M.m11*N.m13+M.m12*N.m23+M.m13*N.m33 ,
          M.m21*N.m11+M.m22*N.m21+M.m23*N.m31, M.m21*N.m12+M.m22*N.m22+M.m23*N.m32, M.m21*N.m13+M.m22*N.m23+M.m23*N.m33 ,
          M.m31*N.m11+M.m32*N.m21+M.m33*N.m31, M.m31*N.m12+M.m32*N.m22+M.m33*N.m32, M.m31*N.m13+M.m32*N.m23+M.m33*N.m33
          };
  #else
  return M*N;
  #endif
#else
  return M*N;
#endif
}


static inline DHCR_DEVICE_FUNC
float4
float4_multiply_float4x4( float4 v,
                          float4x4 M)
{
#if defined(__CUDA_ARCH__)
  return M*v;
#elif defined(__METAL_VERSION__)
  return v*M;
#elif defined(CL_VERSION_1_2)
  #if DEHANCER_GPU_CODE
  return make_float4(
          M.s0*v.x + M.s1*v.y + M.s2*v.z + M.s3*v.w,
          M.s4*v.x + M.s5*v.y + M.s6*v.z + M.s7*v.w,
          M.s8*v.x + M.s9*v.y + M.sA*v.z + M.sB*v.w,
          M.sC*v.x + M.sD*v.y + M.sE*v.z + M.sF*v.w
  );
#else
  return v*M;
#endif
#else
  return (float4){0.000000f, 0.000000f, 0.000000f, 0.000000f};
#endif
}

static inline DHCR_DEVICE_FUNC
float4x4 float4x4_multiply_float4x4( float4x4 M,  float4x4 N)
{
#if defined(__CUDA_ARCH__)
  return M*N;
#elif defined(__METAL_VERSION__)
  return M*N;
#elif defined(CL_VERSION_1_2)
  #if DEHANCER_GPU_CODE
  return (float4x4){
          M.s0*N.s0+M.s1*N.s4+M.s2*N.s8+M.s3*N.sC , M.s0*N.s1+M.s1*N.s5+M.s2*N.s9+M.s3*N.sD , M.s0*N.s2+M.s1*N.s6+M.s2*N.sA+M.s3*N.sE , M.s0*N.s3+M.s1*N.s7+M.s2*N.sB+M.s3*N.sF ,
          M.s4*N.s0+M.s5*N.s4+M.s6*N.s8+M.s7*N.sC , M.s4*N.s1+M.s5*N.s5+M.s6*N.s9+M.s7*N.sD , M.s4*N.s2+M.s5*N.s6+M.s6*N.sA+M.s7*N.sE , M.s4*N.s3+M.s5*N.s7+M.s6*N.sB+M.s7*N.sF ,
          M.s8*N.s0+M.s9*N.s4+M.sA*N.s8+M.sB*N.sC , M.s8*N.s1+M.s9*N.s5+M.sA*N.s9+M.sB*N.sD , M.s8*N.s2+M.s9*N.s6+M.sA*N.sA+M.sB*N.sE , M.s8*N.s3+M.s9*N.s7+M.sA*N.sB+M.sB*N.sF ,
          M.sC*N.s0+M.sD*N.s4+M.sE*N.s8+M.sF*N.sC , M.sC*N.s1+M.sD*N.s5+M.sE*N.s9+M.sF*N.sD , M.sC*N.s2+M.sD*N.s6+M.sE*N.sA+M.sF*N.sE , M.sC*N.s3+M.sD*N.s7+M.sE*N.sB+M.sF*N.sF
  };
#else
  return M*N;
#endif
#else
  return M*N;
#endif
}

#endif //DEHANCER_GPULIB_CMATRIX_OPS_H
