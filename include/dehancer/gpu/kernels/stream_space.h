//
// Created by denn on 08.05.2021.
//

#ifndef DEHANCER_GPULIB_STREAM_SPACE_H
#define DEHANCER_GPULIB_STREAM_SPACE_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/stream_log.h"
#include "dehancer/gpu/kernels/stream_gamma.h"

typedef enum {
    DHCR_Pass        = -1,
    DHCR_ColorSpace  = 0, // rec_709_22, aces, etc...
    DHCR_Camera      = 30
} DHCR_StreamSpace_Type;

typedef struct {
    DHCR_GammaParameters gamma;
    DHCR_LogParameters   log;
} DHCR_StreamSpace_Params;

typedef struct {
    
    bool_t is_identity;
    
    /***
     * Forward transformation matrix from current space to another
     */
    float4x4 cs_forward_matrix;
    
    /***
     * Inverse transformation matrix to current space from another
     */
    float4x4 cs_inverse_matrix;
    
    /***
     * Polynomial and Gama/Log transformation parameters
     */
    DHCR_StreamSpace_Params cs_params;
  
} DHCR_StreamSpace_TransformFunc;

typedef struct {
    bool_t is_identity;
    DHCR_LutParameters forward;
    DHCR_LutParameters inverse;
} DHCR_StreamSpace_TransformLut;

typedef struct _DHCR_StreamSpace_{
    /***
     * Space type
     */
    DHCR_StreamSpace_Type           type;
    
    /***
     * Transformed image can be analyzed and expanded
     */
    bool_t                          expandable;
    
    /***
     * Transform function
     */
    DHCR_StreamSpace_TransformFunc  transform_func;
    
    /***
     * Transform table
     */
    DHCR_StreamSpace_TransformLut   transform_lut;
    
#if __cplusplus && !DEHANCER_GPU_CODE
  
    /***
     * Searchable unique id
     */
    std::string                     id = "rec_709_g22";
    
    /***
     * Name of space can be displayed on UI
     */
    std::string                     name = "Rec.709";
    
    bool operator==(const _DHCR_StreamSpace_ &c) const { return type == c.type && id == c.id; }

#endif

} DHCR_StreamSpace;

static inline DHCR_DEVICE_FUNC
float4x4 stream_matrix_transform_identity() {
#if DEHANCER_GPU_CODE
#if defined(__CUDA_ARCH__)
  float4x4 m; m.setIdentity();
#else
  float4x4 m =
          (float4x4){
                  (float4){1.000000f, 0.000000f, 0.000000f, 0.000000f},
                  (float4){0.000000f, 1.000000f, 0.000000f, 0.000000f},
                  (float4){0.000000f, 0.000000f, 1.000000f, 0.000000f},
                  (float4){0.000000f, 0.000000f, 0.000000f, 1.000000f}
          };
#endif
#else
  float4x4 m =  float4x4(
          {
                  {1.000000f, 0.000000f, 0.000000f, 0.000000f},
                  {0.000000f, 1.000000f, 0.000000f, 0.000000f},
                  {0.000000f, 0.000000f, 1.000000f, 0.000000f},
                  {0.000000f, 0.000000f, 0.000000f, 1.000000f}
          });
#endif
  return m;
}

static inline DHCR_DEVICE_FUNC
DHCR_StreamSpace_TransformFunc stream_space_transform_func_identity() {
  float4x4 m = stream_matrix_transform_identity();
  DHCR_GammaParameters gamma = {false, 0, 0, 0, 0, 0, 0};
  DHCR_LogParameters log = {false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  DHCR_StreamSpace_Params params = { gamma, log };
  DHCR_StreamSpace_TransformFunc identity = {
          true,
          m,
          m,
          params
  };
  return identity;
}

static inline DHCR_DEVICE_FUNC
DHCR_StreamSpace_TransformLut stream_space_transform_lut_identity() {
#if !DEHANCER_GPU_CODE
  static float l[4] = {0.0f,0.0f,0.0f,0.0f};
#endif
  DHCR_LutParameters lut = {
          1, 4, false
#if !DEHANCER_GPU_CODE
          ,l
#endif
  };
  DHCR_StreamSpace_TransformLut identity = {
          true,
          lut,
          lut
  };
  return identity;
}

static inline DHCR_DEVICE_FUNC
DHCR_StreamSpace stream_space_identity() {
  
  DHCR_StreamSpace_TransformFunc func = stream_space_transform_func_identity();
  
  DHCR_StreamSpace_TransformLut  lut  = stream_space_transform_lut_identity();
  
  DHCR_StreamSpace identity = {
          DHCR_ColorSpace,
          false,
          func,
          lut
  };
  
  return identity;
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
  return make_float4(
          M.s0*v.x + M.s1*v.y + M.s2*v.z + M.s3*v.w,
          M.s4*v.x + M.s5*v.y + M.s6*v.z + M.s7*v.w,
          M.s8*v.x + M.s9*v.y + M.sA*v.z + M.sB*v.w,
          M.sC*v.x + M.sD*v.y + M.sE*v.z + M.sF*v.w
  );
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
  return (float4x4){
          M.s0*N.s0+M.s1*N.s4+M.s2*N.s8+M.s3*N.sC , M.s0*N.s1+M.s1*N.s5+M.s2*N.s9+M.s3*N.sD , M.s0*N.s2+M.s1*N.s6+M.s2*N.sA+M.s3*N.sE , M.s0*N.s3+M.s1*N.s7+M.s2*N.sB+M.s3*N.sF ,
          M.s4*N.s0+M.s5*N.s4+M.s6*N.s8+M.s7*N.sC , M.s4*N.s1+M.s5*N.s5+M.s6*N.s9+M.s7*N.sD , M.s4*N.s2+M.s5*N.s6+M.s6*N.sA+M.s7*N.sE , M.s4*N.s3+M.s5*N.s7+M.s6*N.sB+M.s7*N.sF ,
          M.s8*N.s0+M.s9*N.s4+M.sA*N.s8+M.sB*N.sC , M.s8*N.s1+M.s9*N.s5+M.sA*N.s9+M.sB*N.sD , M.s8*N.s2+M.s9*N.s6+M.sA*N.sA+M.sB*N.sE , M.s8*N.s3+M.s9*N.s7+M.sA*N.sB+M.sB*N.sF ,
          M.sC*N.s0+M.sD*N.s4+M.sE*N.s8+M.sF*N.sC , M.sC*N.s1+M.sD*N.s5+M.sE*N.s9+M.sF*N.sD , M.sC*N.s2+M.sD*N.s6+M.sE*N.sA+M.sF*N.sE , M.sC*N.s3+M.sD*N.s7+M.sE*N.sB+M.sF*N.sF
  };
#else
  return M*N;
#endif
}

static inline DHCR_DEVICE_FUNC
float4 transform( float4 in_, DHCR_StreamSpace space, DHCR_TransformDirection direction) {
  
  float4 out = in_;//make_float4(in_.x, in_.y, in_.z, in_.w);
  
  out = float4_multiply_float4x4(out,
                                 direction == DHCR_Forward
                                 ? space.transform_func.cs_forward_matrix
                                 : space.transform_func.cs_inverse_matrix);
  
  float4 next = out;//make_float4(out.x, out.y, out.z, out.w);
  
  if (direction == DHCR_Forward) {
    
    if (space.transform_func.cs_params.log.enabled) {
      next =  apply_log_forward(next, space.transform_func.cs_params.log);
    }
    
    if (space.transform_func.cs_params.gamma.enabled) {
      next =  apply_gamma_forward(next, space.transform_func.cs_params.gamma);
    }
    
  } else {
    
    if (space.transform_func.cs_params.gamma.enabled) {
      next =  apply_gamma_inverse(next, space.transform_func.cs_params.gamma);
    }
    
    if (space.transform_func.cs_params.log.enabled) {
      next =  apply_log_inverse(next, space.transform_func.cs_params.log);
    }
    
  }
  
  //out = make_float4(next.x, next.y, next.z, next.w);
  
  return next;
};

#endif //DEHANCER_GPULIB_STREAM_SPACE_H