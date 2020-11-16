//
// Created by denn nevera on 10/11/2020.
//

#include "Context.h"

namespace dehancer::metal {

    Context::Context(const void *command_queue):
            command_queue_(static_cast<id<MTLCommandQueue>>( (__bridge id) (void*)command_queue))
    {
    }

    id<MTLCommandQueue> Context::get_command_queue() const {
      return static_cast<id<MTLCommandQueue>>( (__bridge id) command_queue_);
    }

    id<MTLDevice> Context::get_device() const {
      return get_command_queue().device;
    }
}