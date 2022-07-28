//
// Created by denn nevera on 18/11/2020.
//

#include "Memory.h"
#include "dehancer/gpu/Log.h"
#import <Metal/Metal.h>

namespace dehancer::metal {
    
    MemoryHolder::MemoryHolder(const void *command_queue, const void* buffer, size_t length, MemoryDesc::MemFlags flags):
            dehancer::MemoryHolder(),
            Context(command_queue),
            memobj_(nullptr),
            length_(length),
            is_self_allocated_(false)
    {
      
      if (length == 0) {
        throw std::runtime_error("Device memory could not be allocated with size: " + std::to_string(length));
      }
      
      MTLResourceOptions res;
      
      if (flags&MemoryDesc::MemFlags::less_memory){
        res = MTLResourceStorageModePrivate | MTLResourceCPUCacheModeDefaultCache;
      }
      else
      if (has_unified_memory()) {
        #if defined(IOS_SYSTEM)
        res = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
        #else
        res = MTLResourceStorageModeManaged | MTLResourceCPUCacheModeDefaultCache;
        #endif
      }
      else {
        res = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
      }
      
      auto device = static_cast<id<MTLDevice>>( (__bridge id) get_device());
      
      if (buffer)
        memobj_ = [device newBufferWithBytes: buffer length: length options:res];
      else
        memobj_ = [device newBufferWithLength:length options:res];
      
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
      
      length_ = static_cast<id <MTLBuffer>>(memobj_).length;
    }
    
    MemoryHolder::~MemoryHolder() {
      if (memobj_ && is_self_allocated_) {
        [static_cast<id <MTLBuffer>>(memobj_) release];
      }
    }
    
    size_t MemoryHolder::get_length() const {
      return  static_cast<id <MTLBuffer>>(memobj_).length;;
    }
    
    const void *MemoryHolder::get_memory() const {
      return memobj_;
    }
    
    void *MemoryHolder::get_memory() {
      return memobj_;
    }
    
    Error MemoryHolder::get_contents(std::vector<uint8_t> &buffer) const {
      return  get_contents(buffer.data(), static_cast<id <MTLBuffer>>(memobj_).length);
    }
    
    Error MemoryHolder::get_contents (void *buffer, size_t length) const {
      
      if (static_cast<id <MTLBuffer>>(memobj_).contents == nullptr)
        return Error(CommonError::PERMISSIONS_ERROR, error_string("Device memory object is private"));
      
      if (static_cast<id <MTLBuffer>>(memobj_).length>length)
        return Error(CommonError::OUT_OF_RANGE, error_string("Device memory length greater then buffer allocated"));
  
      auto queue = static_cast<id<MTLCommandQueue>>( (__bridge id)get_command_queue());
      id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
  
      #if not defined(IOS_SYSTEM)
      id <MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      [blitEncoder synchronizeResource:reinterpret_cast<id <MTLBuffer>>(memobj_)];
      [blitEncoder endEncoding];
      #endif
  
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      
      memcpy(buffer, reinterpret_cast<id <MTLBuffer>>(memobj_).contents, length);
      
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
