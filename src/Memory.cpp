//
// Created by denn nevera on 18/11/2020.
//

#include "dehancer/gpu/Memory.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "platforms/metal/Memory.h"
#elif defined(DEHANCER_GPU_CUDA)
#include "platforms/cuda/Memory.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "platforms/opencl/Memory.h"
#endif

#ifdef DEHANCER_GPU_PLATFORM

namespace dehancer {

    namespace config {
      bool memory::alloc_host_ptr = false;
    }
    
    Memory dehancer::MemoryHolder::Make(const void *command_queue, const void *buffer, size_t length, MemoryDesc::MemFlags flags) {
      return std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::MemoryHolder>(command_queue, buffer, length, flags);
    }

    Memory MemoryHolder::Make(const void *command_queue, void *device_memory) {
      return std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::MemoryHolder>(command_queue,device_memory);
    }

    Memory MemoryHolder::Make(const void *command_queue, std::vector<uint8_t> buffer) {
      return std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::MemoryHolder>(command_queue, buffer);
    }

    Memory MemoryHolder::Make(const void *command_queue, const void *device_memory) {
      return MemoryHolder::Make(command_queue, (void *)device_memory);
    }

    Memory MemoryHolder::Make(const void *command_queue, size_t length) {
      return MemoryHolder::Make(command_queue, nullptr, length);
    }
    
    Memory MemoryHolder::Make (const void *command_queue, const void* from_memory, const MemoryDesc &desc) {
      if (desc.type == MemoryDesc::MemType::host)
        return dehancer::MemoryHolder::Make(command_queue, nullptr, desc.length);
      return dehancer::MemoryHolder::Make(command_queue, from_memory);
    }
    
    Memory MemoryDesc::make(const void *command_queue, const void* from_memory) const {
      if (type == MemType::host)
        return dehancer::MemoryHolder::Make(command_queue, from_memory, length);
      return dehancer::MemoryHolder::Make(command_queue, from_memory);
    }
}

#endif