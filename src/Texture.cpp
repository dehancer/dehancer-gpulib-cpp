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
      try {
        return std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::TextureHolder>(command_queue,desc,from_memory);
      }
      
      catch (const dehancer::texture::memory_exception &e) {
        dehancer::log::error(true, "TextureHolder::Make: memory error: %s", e.what());
        throw dehancer::texture::memory_exception(e.what());
      }
      
      catch (const std::runtime_error &e) {
        dehancer::log::error(true, "TextureHolder::Make: %s", e.what());
        throw dehancer::texture::memory_exception(e.what());
      }
      
      catch (...) {
        dehancer::log::error(true, "TextureHolder::Make: unknown error");
        throw dehancer::texture::memory_exception("Texture GPU memory allocation error");
      }
    }
    
   
    TextureHolder::~TextureHolder () = default;
    
    Texture TextureDesc::make(const void *command_queue, const float *from_memory) const {
      return dehancer::TextureHolder::Make(command_queue, *this, from_memory);
    }
    
    Texture TextureHolder::Make (const void *command_queue, const TextureDesc &desc, const Memory &memory) {
      return dehancer::Texture();
    }
    
    
    size_t TextureDesc::get_hash () const {
      return
              10000000000 * depth
              +
              10000000 * width
              +
              10000 * height
              +
              1000 * static_cast<size_t>(type)
              +
              100 * static_cast<size_t>(pixel_format)
              +
              10 * mem_flags
              +
              channels;
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