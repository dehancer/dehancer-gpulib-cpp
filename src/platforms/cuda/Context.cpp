//
// Created by denn nevera on 10/11/2020.
//

#include "Context.h"

namespace dehancer::cuda {

    Context::Context(const void *command_queue): command_queue_(command_queue)
    {
    }

    CUstream Context::get_command_queue() const {
      return static_cast<CUstream>((void *) command_queue_);
    }
}