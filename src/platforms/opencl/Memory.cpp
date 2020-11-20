//
// Created by denn nevera on 18/11/2020.
//

#include "Memory.h"

namespace dehancer::opencl {

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

      cl_int ret = 0;

      cl_mem_flags flags =  CL_MEM_READ_WRITE;

      if (buffer) flags|=CL_MEM_COPY_HOST_PTR;

      void *data = reinterpret_cast<void*>((void*)buffer);

      memobj_ = clCreateBuffer(
              get_context(),
              flags,
              length,
              data,
              &ret);

      if ( ret != CL_SUCCESS ) {
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
      memobj_ = reinterpret_cast<cl_mem>(device_memory);

      cl_mem_object_type object_type = 0;

      cl_int  ret = clGetMemObjectInfo(memobj_,CL_MEM_TYPE,sizeof(object_type),&object_type, nullptr);
      if (ret != CL_SUCCESS ) {
        memobj_ = nullptr;
        throw std::runtime_error("Device memory could not bind with device handler object");
      }
      if (object_type!=CL_MEM_OBJECT_BUFFER) {
        memobj_ = nullptr;
        throw std::runtime_error("Device memory is not a memory object");
      }

      ret = clGetMemObjectInfo(memobj_,CL_MEM_SIZE,sizeof(size_t),&length_,nullptr);
      if (ret != CL_SUCCESS ) {
        memobj_ = nullptr;
        throw std::runtime_error("Device memory could not get memory size");
      }
    }

    MemoryHolder::~MemoryHolder() {
      if (memobj_ && is_self_allocated_)
        clReleaseMemObject(memobj_);
    }

    size_t MemoryHolder::get_length() const {
      return length_;
    }

    const void *MemoryHolder::get_memory() const {
      return memobj_;
    }

    void *MemoryHolder::get_memory() {
      return memobj_;
    }

    Error MemoryHolder::get_contents(std::vector<uint8_t> &buffer) const {
      return Error(CommonError::NOT_SUPPORTED);
    }
}
