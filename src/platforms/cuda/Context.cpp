//
// Created by denn nevera on 10/11/2020.
//

#include "Context.h"

namespace dehancer::cuda {

    Context::Context(const void *command_queue):
    command_queue_(command_queue),
    context_(nullptr)
    {
      CHECK_CUDA(cuStreamGetCtx(get_command_queue(), &context_));
      //CHECK_CUDA(cuCtxPushCurrent(function_context_));
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
}