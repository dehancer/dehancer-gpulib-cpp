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
            device_ref_({})
    {
      std::lock_guard lock(mutex_);
      
      auto id = reinterpret_cast<std::size_t>(get_command_queue());
      
      auto it = cache_.find(id);
      
      if (it==cache_.end()) {
        
        CHECK_CUDA(cuStreamGetCtx(get_command_queue(), &device_ref_.context));
        push();
        CHECK_CUDA(cuCtxGetDevice(&device_ref_.device_id));
        pop();
        
        cudaDeviceProp info{};
        get_device_info(info);
        if (info.major >= 5) {
          device_ref_.is_half_texture_allowed = true;
        }
  
        device_ref_.max_device_threads = static_cast<size_t>(info.maxThreadsPerBlock);
        device_ref_.max_threads_dim.x=info.maxThreadsDim[0];
        device_ref_.max_threads_dim.y=info.maxThreadsDim[1];
        device_ref_.max_threads_dim.z=info.maxThreadsDim[2];
  
        device_ref_.max_1d_dim = {static_cast<unsigned int>(info.maxTexture1D), 1, 1};
        device_ref_.max_2d_dim = {static_cast<unsigned int>(info.maxTexture2D[0]),static_cast<unsigned int>(info.maxTexture2D[1]) , 1};
        device_ref_.max_3d_dim = {static_cast<unsigned int>(info.maxTexture3D[0]),static_cast<unsigned int>(info.maxTexture3D[1]) , static_cast<unsigned int>(info.maxTexture3D[3])};
  
        cache_[id] = device_ref_;
        
      }
      else {
        device_ref_ = it->second;
      }
    }
    
    CUstream Context::get_command_queue() const {
      return static_cast<CUstream>((void *) command_queue_);
    }
    
    CUcontext Context::get_command_context () const {
      return device_ref_.context;
    }
    
    void Context::push () const {
      CHECK_CUDA(cuCtxPushCurrent(get_command_context()));
    }
    
    void Context::pop () const {
      CHECK_CUDA(cuCtxPopCurrent(&(device_ref_.context)));
    }
    
    CUdevice Context::get_device_id () const {
      return device_ref_.device_id;
    }
    
    void Context::get_device_info (cudaDeviceProp &info) const {
      cudaGetDeviceProperties(&info, device_ref_.device_id);
    }
    
    void Context::get_mem_info (size_t &total, size_t &free) {
      cudaMemGetInfo( &free, &total );
    }
    
    bool Context::is_half_texture_allowed () const {
      return device_ref_.is_half_texture_allowed;
    }
    
    size_t Context::get_max_threads () const {
      return device_ref_.max_device_threads;
    }
    
    dim3 Context::get_max_threads_dim () const {
      return device_ref_.max_threads_dim;
    }
    
    TextureInfo Context::get_texture_info (TextureDesc::Type texture_type) const {
      
      TextureInfo info{};
  
      dim3 max_size;
      
      switch (texture_type) {
        case TextureDesc::Type::i1d:
          max_size = device_ref_.max_1d_dim;
          break;
        case TextureDesc::Type::i2d:
          max_size = device_ref_.max_2d_dim;
          break;
        case TextureDesc::Type::i3d:
          max_size = device_ref_.max_3d_dim;
          break;
      }
      
      info = {
        .max_width  = max_size.x,
        .max_height = max_size.y,
        .max_depth  = max_size.z
      };
      
      return info;
    }
}