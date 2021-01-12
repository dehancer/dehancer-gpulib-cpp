//
// Created by denn nevera on 10/11/2020.
//

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Log.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "platforms/metal/Texture.h"
#elif defined(DEHANCER_GPU_CUDA)
#include "src/platforms/cuda/Texture.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "platforms/opencl/Texture.h"
#endif

#ifdef DEHANCER_GPU_PLATFORM

namespace dehancer {
    
    Texture TextureHolder::Make(const void *command_queue, const TextureDesc &desc, const float *from_memory) {
      return std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::TextureHolder>(command_queue,desc,from_memory);
    }
    
    Texture TextureDesc::make(const void *command_queue, const float *from_memory) {
      return dehancer::TextureHolder::Make(command_queue, *this, from_memory);
    }
    
    bool operator==(const TextureDesc& lhs, const TextureDesc& rhs){
      return
              lhs.type == rhs.type
              &&
              lhs.mem_flags == rhs.mem_flags
              &&
              lhs.pixel_format == rhs.pixel_format
              &&
              lhs.channels == rhs.channels
              &&
              lhs.width == rhs.width
              &&
              lhs.height == rhs.height
              &&
              lhs.depth == rhs.depth
              ;
    }
    
    bool operator!=(const TextureDesc& lhs, const TextureDesc& rhs) {
      return !(lhs==rhs);
    }
}

#endif