//
// Created by denn on 08.05.2021.
//

#ifndef DEHANCER_GPULIB_STREAM_SPACE_H
#define DEHANCER_GPULIB_STREAM_SPACE_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/types.h"

typedef enum {
    DHCR_Pass        = -1,
    DHCR_ColorSpace  = 0, // rec_709_22, aces, etc...
    DHCR_Camera      = 30
} DHCR_StreamSpace_Type;

typedef struct {
    DHCR_GammaParameters gama;
    DHCR_LogParameters   log;
} DHCR_StreamSpace_Params;

typedef struct {
    
    bool is_identity;
    
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
    bool is_identity;
    DHCR_LutParameters forward;
    DHCR_LutParameters inverse;
} DHCR_StreamSpace_TransformLut;

typedef struct {
    /***
     * Space type
     */
    DHCR_StreamSpace_Type           type;
    
    /***
     * Transformed image can be analyzed and expanded
     */
    bool                       expandable;
    
    /***
     * Transform function
     */
    DHCR_StreamSpace_TransformFunc  transform_func;
    
    /***
     * Transform table
     */
    DHCR_StreamSpace_TransformLut   transform_lut;

#if !DEHANCER_GPU_CODE
    /***
     * Searchable unique id
     */
    std::string                       id = "rec_709_g22";
    
    /***
     * Name of space can be displayed on UI
     */
    std::string                     name = "Rec.709";
#endif

} DHCR_StreamSpace;

static inline DHCR_DEVICE_FUNC
        DHCR_StreamSpace_TransformFunc stream_space_transform_func_identity() {
  #if DEHANCER_GPU_CODE
  float4x4 m =
          (float4x4){
                  (float4){1.000000f, 0.000000f, 0.000000f, 0.000000f},
                  (float4){0.000000f, 1.000000f, 0.000000f, 0.000000f},
                  (float4){0.000000f, 0.000000f, 1.000000f, 0.000000f},
                  (float4){0.000000f, 0.000000f, 0.000000f, 1.000000f}
          };
  #else
  float4x4 m = float4x4(
          {
                  {1.000000f, 0.000000f, 0.000000f, 0.000000f},
                  {0.000000f, 1.000000f, 0.000000f, 0.000000f},
                  {0.000000f, 0.000000f, 1.000000f, 0.000000f},
                  {0.000000f, 0.000000f, 0.000000f, 1.000000f}
          });
  #endif
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
  DHCR_LutParameters lut = {
          1, 4, false
          #if !DEHANCER_GPU_CODE
          {0,0,0,0};
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
  
  DHCR_StreamSpace_TransformFunc func = {};
  
  DHCR_StreamSpace_TransformLut  lut  = stream_space_transform_lut_identity();
  
  DHCR_StreamSpace identity = {
          DHCR_ColorSpace,
          false,
          func,
          lut
  };
  
  return identity;
}

#endif //DEHANCER_GPULIB_STREAM_SPACE_H
