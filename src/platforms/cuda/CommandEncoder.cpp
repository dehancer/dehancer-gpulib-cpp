//
// Created by denn nevera on 10/11/2020.
//

#include <dehancer/gpu/CommandEncoder.h>

#include "CommandEncoder.h"

namespace dehancer::cuda {

    CommandEncoder::CommandEncoder(CUfunction kernel,dehancer::cuda::Function* function): kernel_(kernel), function_(function){}

    void CommandEncoder::resize_at_index(int index) {
      if (args_.empty()) {
        args_.resize(index+1, nullptr);
      }
      if (index>=args_.size()) {
        std::vector<void *> old(args_);
        args_.resize(index+1, nullptr);
      }
    }

    void CommandEncoder::set(const Texture &texture, int index)  {
      resize_at_index(index);
      args_.at(index) = texture->get_memory();
    }

    void CommandEncoder::set(const void *bytes, size_t bytes_length, int index)  {

    }

    void CommandEncoder::set(const Memory &memory, int index) {
      resize_at_index(index);
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

    void CommandEncoder::set(const float2x2& m, int index){
    };

    void CommandEncoder::set(const float4x4& m, int index){
    };
}
