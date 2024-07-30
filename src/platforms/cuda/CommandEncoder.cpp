//
// Created by denn nevera on 10/11/2020.
//

#include "dehancer/gpu/CommandEncoder.h"
#include "dehancer/gpu/Lib.h"
//#include "dehancer/gpu/kernels/cuda/cmatrix.h"
#include "CommandEncoder.h"

namespace dehancer::cuda {

    CommandEncoder::CommandEncoder(const CUfunction kernel, const dehancer::cuda::Function* function): kernel_(kernel), function_((dehancer::cuda::Function*)function){}

    void CommandEncoder::resize_at_index(int index) {
      if (args_.empty()) {
        args_.resize(index+1, nullptr);
      }
      if (index >0 && (size_t)index>=args_.size()) {
        std::vector<void *> old(args_);
        args_.resize(index+1, nullptr);
      }
    }

    void CommandEncoder::set(const Texture &texture, int index)  {
      resize_at_index(index);
      #if defined(DEBUG)
      auto m = static_cast<dehancer::nvcc::texture*>(texture->get_memory());
      m->set_label( function_->get_name());
      #endif
      args_.at(index) = texture->get_memory();
    }

    void CommandEncoder::set(const void *bytes, size_t bytes_length, int index)  {
      resize_at_index(index);
      
      std::shared_ptr<char> a = std::shared_ptr<char>(new char[bytes_length]);
      args_container_.emplace_back(a);
  
      memcpy(a.get(),bytes,bytes_length);
      
      args_.at(index) = a.get();
      
    }

    void CommandEncoder::set(const Memory &memory, int index) {
      resize_at_index(index);
      if (memory)
        args_.at(index) = memory->get_pointer();
    }

    void CommandEncoder::set(float p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<decltype(p)>(p); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }

    void CommandEncoder::set(const float2 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::float2>((::float2){p.x(),p.y()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }

    void CommandEncoder::set(bool p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<decltype(p)>(p); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }

    void CommandEncoder::set(int p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<decltype(p)>(p); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }

    void CommandEncoder::set(const float3 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::float3>((::float3){p.x(),p.y(),p.z()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }

    void CommandEncoder::set(const float4 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::float4>((::float4){p.x(),p.y(),p.z(),p.w()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }

    ///
    ///
    /// TODO: float2x2 CommandEncoder!
    ///
    /// \param m
    /// \param index
    union encode_float2x2 {
        struct {
            float m11; float m12;
            float m21; float m22;
        };
        float entries[4];
        float entries2[2][2];
    };

    void CommandEncoder::set(const float2x2& m, int index){
      resize_at_index(index);
      encode_float2x2 data{};
      for (size_t i = 0; i < m.size(); ++i) data.entries[i]=m[i];
      auto a = std::make_shared<encode_float2x2>(data); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }

    union encode_float3x3 {
        struct {
            float m11; float m12; float m13;
            float m21; float m22; float m23;
            float m31; float m32; float m33;
        };
        float entries[9];
        float entries2[3][3];
    };

    void CommandEncoder::set(const float3x3& m, int index){
      resize_at_index(index);
      encode_float3x3 data{};
      for (size_t i = 0; i < m.size(); ++i) data.entries[i]=m[i];
      auto a = std::make_shared<encode_float3x3>(data); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    };
    
    union encode_float4x4 {
        struct {
            float m11; float m12; float m13; float m14;
            float m21; float m22; float m23; float m24;
            float m31; float m32; float m33; float m34;
            float m41; float m42; float m43; float m44;
        };
        float entries[16];
        float entries2[4][4];
    };
    
    void CommandEncoder::set(const float4x4& m, int index){
      resize_at_index(index);
      encode_float4x4 data{};
      for (size_t i = 0; i < m.size(); ++i) data.entries[i]=m[i];
      auto a = std::make_shared<encode_float4x4>(data); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    };
    
    
    
    void CommandEncoder::set(const math::uint2 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::uint2>((::uint2){p.x(),p.y()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    void CommandEncoder::set(const math::uint3 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::uint3>((::uint3){p.x(),p.y(),p.z()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    void CommandEncoder::set(const math::uint4 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::uint4>((::uint4){p.x(),p.y(),p.z(),p.w()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    void CommandEncoder::set(const math::int2 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::int2>((::int2){p.x(),p.y()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    void CommandEncoder::set(const math::int3 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::int3>((::int3){p.x(),p.y(),p.z()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    void CommandEncoder::set(const math::int4 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::int4>((::int4){p.x(),p.y(),p.z(),p.w()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    
    void CommandEncoder::set(const math::bool2 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<uint2>((::uint2){p.x(),p.y()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    void CommandEncoder::set(const math::bool3 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::uint3>((::uint3){p.x(),p.y(),p.z()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    void CommandEncoder::set(const math::bool4 &p, int index) {
      resize_at_index(index);
      auto a = std::make_shared<::uint4>((::uint4){p.x(),p.y(),p.z(),p.w()}); args_container_.emplace_back(a);
      args_.at(index) = a.get();
    }
    
    size_t CommandEncoder::get_block_max_size () const {
      return function_->get_command()->get_max_threads();
    };
    
    CommandEncoder::ComputeSize CommandEncoder::ask_compute_size (size_t width, size_t height, size_t depth) const {
      cudaDeviceProp props{}; function_->get_command()->get_device_info(props);
      auto execution_width = static_cast<size_t>(props.maxBlocksPerMultiProcessor);
      
      ComputeSize compute_size {};
      
      size_t  gsize[2] = {execution_width,execution_width};

      compute_size.block.width  = gsize[0];
      compute_size.block.height = gsize[1];
      
      compute_size.grid.width = ((width + gsize[0] - 1) / gsize[0]);
      compute_size.grid.height = ((height + gsize[1] - 1) / gsize[1]);
      
      compute_size.threads_in_grid = compute_size.grid.width * compute_size.grid.height;
      
      compute_size.grid.depth = depth;
      compute_size.block.depth = 1;
      
      return compute_size;
    }
  
}
