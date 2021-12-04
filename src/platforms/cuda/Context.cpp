//
// Created by denn nevera on 10/11/2020.
//

#include "Context.h"
#include <iostream>

namespace dehancer::cuda {
    
    std::mutex Context::mutex_{};
    std::map<size_t,Context::device_ref> Context::cache_{};
    
    Context::Context(const void *command_queue):
    command_queue_(command_queue),
    context_(nullptr),
    device_id_(0),
    is_half_texture_allowed_(false)
    {
      std::lock_guard lock(mutex_);

      auto id = reinterpret_cast<std::size_t>(get_command_queue());

      auto it = cache_.find(id);

      if (it==cache_.end()) {
  
        CHECK_CUDA(cuStreamGetCtx(get_command_queue(), &context_));
        push();
        CHECK_CUDA(cuCtxGetDevice(&device_id_));
        pop();
        cudaDeviceProp info{};
        get_device_info(info);
        if (info.major >= 7) {
          is_half_texture_allowed_ = true;
        }
  
      device_ref d = {
                context_,
                device_id_,
                is_half_texture_allowed_
        };
        
        cache_[id] = d;
        
      }
      else {
        context_ = it->second.context;
        device_id_ = it->second.device_id;
        is_half_texture_allowed_ = it->second.is_half_texture_allowed;
      }
    }

    CUstream Context::get_command_queue() const {
      return static_cast<CUstream>((void *) command_queue_);
    }
    
    CUcontext Context::get_command_context () const {
      return context_;
    }
    
    void Context::push () const {
      CHECK_CUDA(cuCtxPushCurrent(get_command_context()));
    }
    
    void Context::pop () const {
      CHECK_CUDA(cuCtxPopCurrent(&context_));
    }
    
    CUdevice Context::get_device_id () const {
      return device_id_;
    }
    
    void Context::get_device_info (cudaDeviceProp &info) const {
      cudaGetDeviceProperties(&info, device_id_);
    }
    
    void Context::get_mem_info (size_t &total, size_t &free) {
      cudaMemGetInfo( &free, &total );
    }
    
    bool Context::is_half_texture_allowed () const {
      return is_half_texture_allowed_;
    }
}