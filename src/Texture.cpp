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

#include "dehancer/gpu/Function.h"

namespace dehancer {
    
    Texture TextureHolder::Make(const void *command_queue, const TextureDesc &desc, const float *from_memory, bool is_device_buffer) {
      try {
        return std::move(std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::TextureHolder>(command_queue,desc,from_memory,is_device_buffer));
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
    
    Texture TextureHolder::Make (const void *command_queue, const void *from_memory) {
      try {
        return std::move(std::make_shared<dehancer::DEHANCER_GPU_PLATFORM::TextureHolder>(command_queue,from_memory));
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
    
    Texture TextureHolder::Crop (const Texture &texture,
                                 float left, float right, float top, float bottom,
                                 TextureDesc::PixelFormat format
                                 ) {
      
      auto desc = texture->get_desc();
  
      desc.pixel_format = format;
      
      int origin_left = (int)(float(desc.width)  * left);
      int origin_top  = (int)(float(desc.height)  * top);

      desc.width = (size_t)(float(desc.width)  * (1.0f - left - right));
      desc.height = (size_t)(float(desc.height)  * (1.0f - top - bottom));

      if (desc.width<=0) return nullptr;
      if (desc.height<=0) return nullptr;

      /***
       * TODO:
       * add Function from kernel source code
       */
      auto function = dehancer::Function(texture->get_command_queue(),
                                                           "kernel_crop");
      
      auto result = desc.make(texture->get_command_queue());

      function.execute([&texture,&result,origin_left,origin_top](dehancer::CommandEncoder &command_encoder) {
          command_encoder.set(texture, 0);
          command_encoder.set(result, 1);
          command_encoder.set(origin_left,2);
          command_encoder.set(origin_top,3);
          return dehancer::CommandEncoder::Size::From(result);
      });

      return std::move(result);
    }
    
//    Texture TextureHolder::make_cropped (float left, float right, float top, float bottom) const {
//      Texture source_texture = TextureHolder::Make(this->get_command_queue(), this->get_memory());
//      return std::move(Crop(source_texture, left, right, top, bottom));
//    }
    
    TextureHolder::~TextureHolder () = default;
    
    Texture TextureDesc::make(const void *command_queue, const float *from_memory) const {
      return std::move(dehancer::TextureHolder::Make(command_queue, *this, from_memory));
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