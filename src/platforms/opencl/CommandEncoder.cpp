//
// Created by denn nevera on 10/11/2020.
//

#include <dehancer/gpu/CommandEncoder.h>

#include "CommandEncoder.h"

namespace dehancer::opencl {

    CommandEncoder::CommandEncoder(cl_kernel kernel,dehancer::opencl::Function* function): kernel_(kernel), function_(function){}

    void CommandEncoder::set(const Texture &texture, int index)  {
      if (kernel_) {
        auto memobj = static_cast<cl_mem>(texture->get_memory());
        auto ret = clSetKernelArg(kernel_, index, sizeof(cl_mem), (void *)&memobj);
        if (ret != CL_SUCCESS)
          throw std::runtime_error("Unable to pass to kernel "+function_->get_name()+" the texture buffer at index: " + std::to_string(index));
      }
      else {
        throw std::runtime_error("Unable to pass texture to null kernel "+function_->get_name());
      }
    }

    void CommandEncoder::set(const void *bytes, size_t bytes_length, int index)  {
      if (kernel_){
        auto ret = clSetKernelArg(kernel_, index, bytes_length,  bytes);
        if (ret != CL_SUCCESS)
          throw std::runtime_error("Unable to pass to kernel "+function_->get_name()+" bytes at index: " + std::to_string(index));
      }
      else {
        throw std::runtime_error("Unable to pass bytes to null kernel "+function_->get_name());
      }
    }

    void CommandEncoder::set(const Memory &memory, int index) {
      if (kernel_) {
        auto memobj = static_cast<cl_mem>(memory->get_memory());
        auto ret = clSetKernelArg(kernel_, index, sizeof(cl_mem), (void *)&memobj);
        if (ret != CL_SUCCESS)
          throw std::runtime_error("Unable to pass to kernel "+function_->get_name()+" the memory object at index: " + std::to_string(index));
      }
      else {
        throw std::runtime_error("Unable to pass memory to null kernel "+function_->get_name());
      }
    }

    void CommandEncoder::set(const float4 &p, int index) {
      cl_float4 buf = { p.x(), p.y(), p.z(), p.w()};
      set(&buf, sizeof(buf), index);
    }

    void CommandEncoder::set(const float3 &p, int index) {
      cl_float3 buf = { p.x(), p.y(), p.z()};
      set(&buf, sizeof(buf), index);
    }

    void CommandEncoder::set(float p, int index) {
      cl_float buf = p;
      set(&buf, sizeof(buf), index);
    }

    void CommandEncoder::set(const float2 &p, int index) {
      cl_float2 buf = { p.x(), p.y()};
      set(&buf, sizeof(buf), index);
    }

    void CommandEncoder::set(bool p, int index) {
      cl_bool buf = p;
      set(&buf, sizeof(buf), index);
    }

    void CommandEncoder::set(const float2x2& m, int index){
      cl_float4 mat;
      for (int i = 0; i < m.size(); ++i) mat.s[i]=m[i];
      set(&mat, sizeof(mat), index);
    };
    
    void CommandEncoder::set(const float3x3& m, int index){
      cl_float mat[9];
      for (int i = 0; i < m.size(); ++i) mat[i]=m[i];
      set(&mat, sizeof(mat), index);
    };
    
    void CommandEncoder::set(const float4x4& m, int index){
      cl_float16 mat;
      for (int i = 0; i < m.size(); ++i) mat.s[i]=m[i];
      set(&mat, sizeof(mat), index);
    };
    
    void CommandEncoder::set(const math::uint4 &p, int index) {
      cl_uint4 buf = { p.x(), p.y(), p.z(), p.w()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::uint3 &p, int index) {
      cl_uint3 buf = { p.x(), p.y(), p.z()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::uint2 &p, int index) {
      cl_uint2 buf = { p.x(), p.y()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::int4 &p, int index) {
      cl_int4 buf = { p.x(), p.y(), p.z(), p.w()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::int3 &p, int index) {
      cl_int3 buf = { p.x(), p.y(), p.z()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::int2 &p, int index) {
      cl_int2 buf = { p.x(), p.y()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::bool4 &p, int index) {
      cl_uint4 buf = { p.x(), p.y(), p.z(), p.w()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::bool3 &p, int index) {
      cl_uint3 buf = { p.x(), p.y(), p.z()};
      set(&buf, sizeof(buf), index);
    }
    
    void CommandEncoder::set(const math::bool2 &p, int index) {
      cl_uint2 buf = { p.x(), p.y()};
      set(&buf, sizeof(buf), index);
    }
  
}
