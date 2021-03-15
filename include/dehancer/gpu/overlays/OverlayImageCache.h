//
// Created by denn nevera on 2019-12-24.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/TextureInput.h"
#include "dehancer/gpu/StreamSpace.h"

#include <mutex>

namespace dehancer::overlay {
        
        enum class Resolution : int {
            Default = 2,
            R1K = 1,
            R4K = 2,
            R8K = 3
        };
        
        static inline Resolution make_from (const Texture &source) {
          
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
        
        class image_cache {
        public:
            
            typedef std::unordered_map<size_t, std::shared_ptr<dehancer::TextureInput>> OverlayImages;
        
            Texture get(const void *command_queue,
                        overlay::Resolution resolution,
                        const std::string &file,
                        const StreamSpace &space = StreamSpace());

            Texture get(const void *command_queue,
                        overlay::Resolution resolution,
                        const uint8_t *image_buffer,
                        size_t length,
                        const StreamSpace &space = StreamSpace());
    
            Texture get(const void *command_queue,
                        overlay::Resolution resolution,
                        const std::vector<uint8_t>& buffer,
                        const StreamSpace &space = StreamSpace());
            
            explicit image_cache(size_t cache_index);
            
            ~image_cache();
        
        private:
            OverlayImages cache_;
            mutable std::mutex mutex_;
            size_t cache_index_;
        };
        
        /***
         * Common singleton interface
         * @tparam T
         */
        template<typename T, size_t N>
        class ImageCache {
        public:
            static T &Instance() {
              static T instance(N);
              return instance;
            }
        
        protected:
            ImageCache() = default;
            ~ImageCache() = default;
        
        public:
            ImageCache(ImageCache const &) = delete;
            ImageCache &operator=(ImageCache const &) = delete;
        };
      
    }