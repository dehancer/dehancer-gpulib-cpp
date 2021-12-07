//
// Created by denn nevera on 18/11/2020.
//

#include "Memory.h"
#include "utils.h"

namespace dehancer::cuda {

    MemoryHolder::MemoryHolder(const void *command_queue, const void* buffer, size_t length, MemoryDesc::MemFlags flags):
            dehancer::MemoryHolder(),
            Context(command_queue),
            memobj_(0),
            length_(length),
            is_self_allocated_(false)
    {
  
      push();
      
      if (length == 0) {
        pop();
        throw std::runtime_error("Device memory could not be allocated with size: " + std::to_string(length));
      }
      is_self_allocated_ = true;

      CHECK_CUDA(cudaMalloc((void**)&memobj_, length_));
      
      if (buffer) {
        auto *p = static_cast<uint8_t*>((void*)buffer);
        CHECK_CUDA(cudaMemcpyAsync((void*)memobj_, p, length, cudaMemcpyHostToDevice, get_command_queue()));
      }
      pop();
    }

    MemoryHolder::MemoryHolder(const void *command_queue, std::vector<uint8_t> buffer):
            MemoryHolder(command_queue,buffer.data(),buffer.size())
    {}

    MemoryHolder::MemoryHolder(const void *command_queue, void *device_memory):
            dehancer::MemoryHolder(),
            Context(command_queue),
            memobj_(0),
            length_(0),
            is_self_allocated_(false)
    {
      push();
      memobj_ = reinterpret_cast<CUdeviceptr>(device_memory);
      CUdeviceptr pbase;
      CHECK_CUDA(cuMemGetAddressRange (&pbase, &length_, memobj_ ));
      pop();
    }

    MemoryHolder::~MemoryHolder() {
      if (is_self_allocated_ && memobj_) {
        push();
        cudaFree((void *) memobj_);
        pop();
      }
    }

    size_t MemoryHolder::get_length() const {
      return length_;
    }

    const void *MemoryHolder::get_memory() const {
      return reinterpret_cast<const void *>(memobj_);
    }

    void *MemoryHolder::get_memory() {
      return reinterpret_cast<void *>(memobj_);
    }

    const void *MemoryHolder::get_pointer() const {
      if (memobj_)
        return &memobj_;
      return nullptr;
    }

    void *MemoryHolder::get_pointer() {
      if (memobj_)
        return &memobj_;
      return nullptr;
    }
    Error MemoryHolder::get_contents(std::vector<uint8_t> &buffer) const {
      buffer.resize( get_length());
      return get_contents(buffer.data(), get_length());
    }

    Error MemoryHolder::get_contents(void *buffer, size_t length) const {
      if (!memobj_)
        return Error(CommonError::OUT_OF_RANGE, "Memory object is null");

      if (length<get_length())
        return Error(CommonError::OUT_OF_RANGE, "Buffer length not enough to copy memory object");

      try {
        this->push();
        CHECK_CUDA(cudaMemcpyAsync(buffer, (const void *)memobj_, get_length(), cudaMemcpyDeviceToHost, get_command_queue()));
        this->pop();
      }
      catch (const std::runtime_error &e) {
        return Error(CommonError::EXCEPTION, e.what());
      }

      return Error(CommonError::OK);
    }
}
