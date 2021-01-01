//
// Created by denn nevera on 18/11/2020.
//

#include "Memory.h"

namespace dehancer::cuda {

    MemoryHolder::MemoryHolder(const void *command_queue, const void* buffer, size_t length):
            dehancer::MemoryHolder(),
            Context(command_queue),
            //memobj_(nullptr),
            length_(length),
            is_self_allocated_(false)
    {
      is_self_allocated_ = true;
    }

    MemoryHolder::MemoryHolder(const void *command_queue, std::vector<uint8_t> buffer):
            MemoryHolder(command_queue,buffer.data(),buffer.size())
    {}

    MemoryHolder::MemoryHolder(const void *command_queue, void *device_memory):
            dehancer::MemoryHolder(),
            Context(command_queue),
            //memobj_(nullptr),
            length_(0),
            is_self_allocated_(false)
    {
    }

    MemoryHolder::~MemoryHolder() {
    }

    size_t MemoryHolder::get_length() const {
      return length_;
    }

    const void *MemoryHolder::get_memory() const {
      return nullptr;
    }

    void *MemoryHolder::get_memory() {
      return nullptr;
    }

    Error MemoryHolder::get_contents(std::vector<uint8_t> &buffer) const {
      return Error(CommonError::NOT_SUPPORTED);
    }
}
