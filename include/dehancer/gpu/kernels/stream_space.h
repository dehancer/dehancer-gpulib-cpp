//
// Created by denn on 08.05.2021.
//

#ifndef DEHANCER_GPULIB_STREAM_SPACE_H
#define DEHANCER_GPULIB_STREAM_SPACE_H

#include "dehancer/gpu/kernels/cmatrix_ops.h"
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

typedef struct
#if defined(CL_VERSION_1_2) || defined(DEHANCER_GPU_OPENCL)
__attribute__ ((packed))
#endif
{
    
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
          (bool_t)true,
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
          (bool_t)false,
          1,
          4
#if !DEHANCER_GPU_CODE
          ,l
#endif
  };
  DHCR_StreamSpace_TransformLut identity = {
          (bool_t)true,
          lut,
          lut
  };
  return identity;
}

typedef struct _DHCR_StreamSpace_ {
    /***
     * Space type
     */
    DHCR_StreamSpace_Type           type
    #if __cplusplus && !DEHANCER_GPU_CODE
    = DHCR_ColorSpace;
    #else
    ;
    #endif
    
    /***
     * Transformed image can be analyzed and expanded
     */
    bool_t                          expandable
    #if __cplusplus && !DEHANCER_GPU_CODE
    = (bool_t)false;
    #else
    ;
    #endif
    
    /***
     * Transform function
     */
    DHCR_StreamSpace_TransformFunc  transform_func
    #if __cplusplus && !DEHANCER_GPU_CODE
    = stream_space_transform_func_identity();
    #else
    ;
    #endif
    
    /***
     * Transform table
     */
    DHCR_StreamSpace_TransformLut   transform_lut
    #if __cplusplus && !DEHANCER_GPU_CODE
    = stream_space_transform_lut_identity();
    #else
    ;
    #endif
    
#if __cplusplus && !DEHANCER_GPU_CODE
  
    /***
     * Searchable unique id
     */
    std::string                     id = "rec_709_g22";
    
    /***
     * Name of space can be displayed on UI
     */
    std::string                     name = "Rec.709";
    
    /**
     *
     * deprecated
     *
     */

    bool operator==(const _DHCR_StreamSpace_ &c) const { return type == c.type && id == c.id; }
    
#endif

} DHCR_StreamSpace;


#if __cplusplus && !DEHANCER_GPU_CODE
static inline DHCR_DEVICE_FUNC
DHCR_StreamSpace stream_space_identity() {
  
  DHCR_StreamSpace_TransformFunc func = stream_space_transform_func_identity();
  
  DHCR_StreamSpace_TransformLut  lut  = stream_space_transform_lut_identity();
  
  DHCR_StreamSpace identity = {
          DHCR_ColorSpace,
          (bool_t)false,
          func,
          lut
  };
  
  return identity;
}
#endif

static inline DHCR_DEVICE_FUNC
float4 transform( float4 in_, DHCR_StreamSpace space, DHCR_TransformDirection direction) {
  
  float4 next = in_;
  
//  next = float4_multiply_float4x4(next,
//                                 direction == DHCR_Forward
//                                 ? space.transform_func.cs_forward_matrix
//                                 : space.transform_func.cs_inverse_matrix);
//
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
  
  return next;
};

static inline DHCR_DEVICE_FUNC
float4 transform_extended( float4 in_, DHCR_GammaParameters gamma, DHCR_LogParameters log, DHCR_CONST_ARG float4x4_ref_t cs_forward_matrix, DHCR_CONST_ARG float4x4_ref_t cs_inverse_matrix, DHCR_TransformDirection direction) {
  
  float4 next = in_;

//  next = float4_multiply_float4x4(next,
//                                 direction == DHCR_Forward
//                                 ? cs_forward_matrix
//                                 : cs_inverse_matrix);

  if (direction == DHCR_Forward) {
    
    if (log.enabled) {
      next =  apply_log_forward(next, log);
    }
    
    if (gamma.enabled) {
      next =  apply_gamma_forward(next, gamma);
    }
    
  } else {
    
    if (gamma.enabled) {
      next =  apply_gamma_inverse(next, gamma);
    }
    
    if (log.enabled) {
      next =  apply_log_inverse(next, log);
    }
    
  }
  
  return next;
};

#endif //DEHANCER_GPULIB_STREAM_SPACE_H
