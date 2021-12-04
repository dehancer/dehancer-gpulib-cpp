//
// Created by denn nevera on 10/11/2020.
//

#include "Context.h"

namespace dehancer::cuda {

    Context::Context(const void *command_queue):
    command_queue_(command_queue),
    context_(nullptr),
    device_id_(0)
    {
      CHECK_CUDA(cuStreamGetCtx(get_command_queue(), &context_));
      push();
      CHECK_CUDA(cuCtxGetDevice(&device_id_));
      pop();
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
}