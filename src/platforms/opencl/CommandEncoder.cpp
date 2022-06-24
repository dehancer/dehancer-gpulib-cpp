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
          throw std::runtime_error("Unable to pass to kernel "+function_->get_name()+" bytes at index: " + std::to_string(index) + ", error code: " + std::to_string(ret));
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
    
    void CommandEncoder::set(const float2x2& m, int index){
      cl_float4 mat;
      for (int i = 0; i < (int)m.size(); ++i) mat.s[i]=m[i];
      set(&mat, sizeof(mat), index);
    }
    
    size_t CommandEncoder::get_block_max_size () const {
      size_t workgroup_size;
      clGetKernelWorkGroupInfo(kernel_,
                               function_->get_command()->get_device_id(),
                               CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                               &workgroup_size, nullptr);
      return workgroup_size;
    }
    
    CommandEncoder::ComputeSize CommandEncoder::ask_compute_size (size_t width, size_t height, size_t depth) const {
  
      size_t workgroup_size = get_block_max_size();
      
      ComputeSize compute_size {};
      
      size_t  gsize[2];
  
      if (workgroup_size <= 256)
      {
        gsize[0] = 16;
        gsize[1] = workgroup_size / 16;
      }
      else if (workgroup_size <= 1024)
      {
        gsize[0] = workgroup_size / 16;
        gsize[1] = 16;
      }
      else
      {
        gsize[0] = workgroup_size / 32;
        gsize[1] = 32;
      }
  
      compute_size.block.width  = gsize[0];
      compute_size.block.height = gsize[1];
  
      compute_size.grid.width = ((width + gsize[0] - 1) / gsize[0]);
      compute_size.grid.height = ((height + gsize[1] - 1) / gsize[1]);
  
      compute_size.threads_in_grid = compute_size.grid.width * compute_size.grid.height;
  
      compute_size.grid.width  *= gsize[0];
      compute_size.grid.height *= gsize[1];
  
      compute_size.grid.depth = depth;
      compute_size.block.depth = 1;
      
      return compute_size;
    };
    
    void CommandEncoder::set(const float3x3& m, int index){
      cl_float mat[9];
      for (int i = 0; i < (int)m.size(); ++i) mat[i]=m[i];
      set(&mat, sizeof(mat), index);
    };
    
    void CommandEncoder::set(const float4x4& m, int index){
      cl_float16 mat;
      for (int i = 0; i < (int)m.size(); ++i) mat.s[i]=m[i];
      set(&mat, sizeof(mat), index);
    }
    
   
    void CommandEncoder::set (const dehancer::StreamSpace &p, int index) {
      StreamSpace copy = p;
      set(&copy, sizeof(copy), index);
    }
}
