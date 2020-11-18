//
// Created by denn nevera on 18/11/2020.
//

#include "dehancer/gpu/Memory.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "platforms/metal/Memory.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "platforms/opencl/Memory.h"
#endif

namespace dehancer {

    Memory dehancer::MemoryHolder::Make(const void *command_queue, const void *buffer, size_t length) {
      return std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::MemoryHolder>(command_queue, buffer, length);
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
}