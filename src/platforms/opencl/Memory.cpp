//
// Created by denn nevera on 18/11/2020.
//

#include "Memory.h"

namespace dehancer::opencl {

    MemoryHolder::MemoryHolder(const void *command_queue, const void* buffer, size_t length, MemoryDesc::MemFlags mflags):
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
      
      if (config::memory::alloc_host_ptr)
        flags |= CL_MEM_ALLOC_HOST_PTR;

      if (buffer) flags |= CL_MEM_COPY_HOST_PTR;
      else if (mflags&MemoryDesc::MemFlags::less_memory){
        flags |= CL_MEM_HOST_NO_ACCESS;
      }
      
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
  
      #ifdef PRINT_KERNELS_DEBUG
      //std::cout << "TextureHolder::~TextureHolder(" << memobj_<< "[" << length_ << "]" << " :: " << is_self_allocated_ << ")"  << std::endl;
      #endif
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
      buffer.resize( get_length());
      return get_contents(buffer.data(), get_length());
    }

    Error MemoryHolder::get_contents(void *buffer, size_t length) const {

      if (!memobj_)
        return Error(CommonError::OUT_OF_RANGE, "Memory object is null");

      if (length<get_length())
        return Error(CommonError::OUT_OF_RANGE, "Buffer length not enough to copy memory object");

      auto ret = clEnqueueReadBuffer(get_command_queue(),
                                     memobj_,
                                     CL_TRUE,
                                     0,
                                     get_length(),
                                     buffer,
                                     0,
                                     nullptr,
                                     nullptr);

      if (ret != CL_SUCCESS) {
        return Error(CommonError::EXCEPTION, "Memory could not be read");
      }

      return Error(CommonError::OK);
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

}
