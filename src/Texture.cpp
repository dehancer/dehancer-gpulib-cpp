//
// Created by denn nevera on 10/11/2020.
//

#include "dehancer/gpu/Texture.h"
#include "opencl/OCLTexture.h"

namespace dehancer {

    Texture TextureHolder::Make(const void *command_queue, const TextureDesc &desc) {
      return std::make_shared<dehancer::opencl::TextureHolder>(command_queue,desc);
    }
}
