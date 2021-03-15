//
// Created by denn nevera on 2019-12-24.
//

#include "dehancer/gpu/overlays/OverlayImageCache.h"
#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/Log.h"

#include <string>
#include <iostream>

namespace dehancer::overlay {
    
    Resolution resolution_from (const Texture &source) {
      
      Resolution resolution;
      
      if (source->get_width() > 4096) {
        
        resolution = Resolution::R8K;
        
      } else if (source->get_width() > 1024) {
        
        resolution = Resolution::R4K;
        
      } else {
        
        resolution = Resolution::R1K;
        
      }
      
      return resolution;
    }
    
    image_cache::image_cache(size_t cache_index):
            cache_(),
            cache_index_(cache_index) {
    }

    Texture  image_cache::get(const void *command_queue,
                                     overlay::Resolution resolution,
                                     const std::string &file,
                                     const StreamSpace &space) const {

        std::unique_lock<std::mutex> lock(mutex_);

        size_t file_hash = cache_index_
                           + static_cast<size_t>(resolution)
                           + reinterpret_cast<size_t>(command_queue)
                           + static_cast<size_t>(space.type) * 1000;

        const auto it = cache_.find(file_hash);

        if (it == cache_.end()) {

            try {
                auto t = std::make_shared<dehancer::TextureInput>(command_queue);

                if (t) {
                    std::ifstream ifs(file, std::ios::binary);
                    ifs >> *t;

                    cache_[file_hash] = t;

                    return t->get_texture();
                }
                return nullptr;
            }
            catch (const std::exception &e) {
#ifdef PRINT_DEBUG
                dehancer::log::error(true, " **** overlay cache error: %i: file = %s is not VALID. message = %s", file_hash, file.c_str(), e.what());
#endif
            }
        }
        else {
            return it->second->get_texture();
        }

        return nullptr;
    }
    
    Texture
    image_cache::get (const void *command_queue,
                      overlay::Resolution resolution,
                      const uint8_t *image_buffer,
                      size_t length,
                      const StreamSpace &space) const {
      if (!image_buffer) return nullptr;
      return get(command_queue,resolution,std::vector<uint8_t>(image_buffer,image_buffer+length),space);
    }
    
    Texture
    image_cache::get (const void *command_queue, overlay::Resolution resolution, const std::vector<uint8_t> &buffer,
                      const StreamSpace &space) const {
      
      std::unique_lock<std::mutex> lock(mutex_);
  
      size_t file_hash = cache_index_
                         + static_cast<size_t>(resolution)
                         + reinterpret_cast<size_t>(command_queue)
                         + static_cast<size_t>(space.type) * 1000;
  
      const auto it = cache_.find(file_hash);
  
      if (it == cache_.end()) {
    
        try {
          
          auto t = std::make_shared<dehancer::TextureInput>(command_queue);
      
          if (t) {
  
            if (buffer.empty()) return nullptr;
            
            t->load_from_image(buffer);
        
            cache_[file_hash] = t;
        
            return t->get_texture();
            
          }
          
          return nullptr;
          
        }
        catch (const std::exception &e) {
#ifdef PRINT_DEBUG
          dehancer::log::error(true, " **** overlay cache error: %i: image buffer is not VALID. message = %s", file_hash, e.what());
#endif
        }
      }
      else {
        return it->second->get_texture();
      }
  
      return nullptr;
    }
    
    Texture image_cache::get (const void *command_queue, overlay::Resolution resolution) const {
      return  get(command_queue, resolution, StreamSpace());
    }
    
    Texture image_cache::get (const void *command_queue) const {
      return get(command_queue, Resolution::Default);
    }
    
    image_cache::~image_cache() = default;
}
