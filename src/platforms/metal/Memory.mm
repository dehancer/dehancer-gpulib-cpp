//
// Created by denn nevera on 18/11/2020.
//

#include "Memory.h"

namespace dehancer::metal {

    MemoryHolder::MemoryHolder(const void *command_queue, const void* buffer, size_t length):
            dehancer::MemoryHolder(),
            Context(command_queue),
            memobj_(nullptr),
            length_(length),
            is_self_allocated_(false)
    {

      if (length == 0) {
        throw std::runtime_error("Device memory could not be allocated with size: " + std::to_string(length));
      }

      if (buffer)
        memobj_ = [get_device() newBufferWithBytes: buffer length: length options:MTLResourceStorageModeManaged];
      else
        memobj_ = [get_device() newBufferWithLength:length options:MTLResourceStorageModeManaged];

      if ( !memobj_ ) {
        throw std::runtime_error("Device memory could not be allocated with size: " + std::to_string(length));
      }
      is_self_allocated_ = true;
    }

    MemoryHolder::MemoryHolder(const void *command_queue, std::vector<uint8_t> buffer):
            MemoryHolder(command_queue,buffer.data(),buffer.size())
    {}

    MemoryHolder::MemoryHolder(const void *command_queue, void *device_memory):
            dehancer::MemoryHolder(),
            Context(command_queue),
            memobj_(nullptr),
            length_(0),
            is_self_allocated_(false)
    {
      memobj_ = reinterpret_cast<id<MTLBuffer> >((__bridge id)device_memory);

      if (!memobj_) {
        throw std::runtime_error("Device memory could not bind with device handler object");
      }

      length_ = memobj_.length;
    }

    MemoryHolder::~MemoryHolder() {
      if (memobj_ && is_self_allocated_)
        [memobj_ release];
    }

    size_t MemoryHolder::get_length() const {
      return  memobj_.length;;
    }

    const void *MemoryHolder::get_memory() const {
      return memobj_;
    }

    void *MemoryHolder::get_memory() {
      return memobj_;
    }

    Error MemoryHolder::get_contents(std::vector<uint8_t> &buffer) const {
      if (memobj_.contents == nullptr)
        return Error(CommonError::PERMISSIONS_ERROR, error_string("Device memory object is private"));
      buffer.resize(get_length());
      memcpy(buffer.data(), memobj_.contents, get_length());
      return Error(CommonError::OK);
    }
    
    Error MemoryHolder::get_contents (void *buffer, size_t length) const {
      if (memobj_.contents == nullptr)
        return Error(CommonError::PERMISSIONS_ERROR, error_string("Device memory object is private"));
      if (memobj_.length>length)
        return Error(CommonError::OUT_OF_RANGE, error_string("Device memory length greater then buffer allocated"));
      memcpy(buffer, memobj_.contents, length);
      return Error(CommonError::OK);
    }
    
    const void *MemoryHolder::get_pointer () const {
      if (memobj_) return &memobj_;
      return nullptr;
    }
    
    void *MemoryHolder::get_pointer () {
      if (memobj_) return &memobj_;
      return nullptr;
    }
    
   
}
