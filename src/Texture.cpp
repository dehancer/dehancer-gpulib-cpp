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
}

#endif