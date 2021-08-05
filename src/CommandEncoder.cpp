//
// Created by denn nevera on 20/11/2020.
//

#include "dehancer/gpu/CommandEncoder.h"

namespace dehancer {

    void dehancer::CommandEncoder::set(bool p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(char p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(int8_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(int16_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(int32_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(uint8_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(uint16_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(uint32_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(float p, int index) {
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(const float2 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void CommandEncoder::set(const float3 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void CommandEncoder::set(const float4 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void dehancer::CommandEncoder::set(const float2x2 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }
    
    void dehancer::CommandEncoder::set(const float3x3 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }
    
    void CommandEncoder::set(const float4x4 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }
    
    void dehancer::CommandEncoder::set(const math::uint2 &p, int index) {
      set(p.mem, p.size()*sizeof(uint), index);
    }
    
    void CommandEncoder::set(const math::uint3 &p, int index) {
      set(p.mem, p.size()*sizeof(uint), index);
    }
    
    void CommandEncoder::set(const math::uint4 &p, int index) {
      set(p.mem, p.size()*sizeof(uint), index);
    }
    
    void dehancer::CommandEncoder::set(const math::int2 &p, int index) {
      set(p.mem, p.size()*sizeof(int), index);
    }
    
    void CommandEncoder::set(const math::int3 &p, int index) {
      set(p.mem, p.size()*sizeof(int), index);
    }
    
    void CommandEncoder::set(const math::int4 &p, int index) {
      set(p.mem, p.size()*sizeof(int), index);
    }
    
    void dehancer::CommandEncoder::set(const math::bool2 &p, int index) {
      set(p.mem, p.size()*sizeof(bool), index);
    }
    
    void CommandEncoder::set(const math::bool3 &p, int index) {
      set(p.mem, p.size()*sizeof(bool), index);

    }
    
    void CommandEncoder::set(const math::bool4 &p, int index) {
      set(p.mem, p.size()*sizeof(bool), index);
    }
    
    
    typedef struct __attribute__((packed)) {
        
        bool_t is_identity;
        
        /***
         * Forward transformation matrix from current space to another
         */
        float cs_forward_matrix[16];
        
        /***
         * Inverse transformation matrix to current space from another
         */
        float cs_inverse_matrix[16];
        
        /***
         * Polynomial and Gama/Log transformation parameters
         */
        DHCR_StreamSpace_Params cs_params;
      
    } gpu_DHCR_StreamSpace_TransformFunc;
    
    typedef struct {
        bool_t   enabled;
        uint   size;
        uint   channels;
    } gpu_DHCR_LutParameters;
    
    typedef struct {
        bool_t is_identity;
        gpu_DHCR_LutParameters forward;
        gpu_DHCR_LutParameters inverse;
    } gpu_DHCR_StreamSpace_TransformLut;
    
    
    typedef struct  {
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
        gpu_DHCR_StreamSpace_TransformFunc  transform_func;
        
        /***
         * Transform table
         */
        gpu_DHCR_StreamSpace_TransformLut   transform_lut;
      
    } gpu_DHCR_StreamSpace;
    
    void CommandEncoder::set (const dehancer::StreamSpace &p, int index) {
      gpu_DHCR_StreamSpace space = {
        .type = p.type,
        .expandable = p.expandable,
        .transform_func = {
                .is_identity = p.transform_func.is_identity,
                .cs_params = p.transform_func.cs_params
        },
        .transform_lut = {
                .is_identity = p.transform_lut.is_identity,
        }
      };
  
      memcpy(space.transform_func.cs_forward_matrix, p.transform_func.cs_forward_matrix.mem, sizeof(space.transform_func.cs_forward_matrix));
      memcpy(space.transform_func.cs_inverse_matrix, p.transform_func.cs_inverse_matrix.mem, sizeof(space.transform_func.cs_inverse_matrix));
      memcpy(&space.transform_lut.forward, &p.transform_lut.forward, sizeof(space.transform_lut.forward));
      memcpy(&space.transform_lut.inverse, &p.transform_lut.inverse, sizeof(space.transform_lut.inverse));
      
      set(&space,sizeof(space),index);
    }
}