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
        Default = 0,
        LandscapeR1K = 0,
        LandscapeR4K = 1,
        LandscapeR8K = 2,
        PortraitR1K = 3,
        PortraitR4K = 4,
        PortraitR8K = 5
    };
    
    struct ItemInfo {
        Resolution  resolution;
        std::string name;
        uint8_t*    buffer;
        size_t      length;
    };
    
    Resolution resolution_from (const Texture &source);
    
    class image_cache {
    
    public:
        
        typedef std::unordered_map<size_t, std::shared_ptr<dehancer::TextureInput>> OverlayImages;
    
        virtual const std::vector<ItemInfo>& available() const = 0;
        
        virtual Texture get(const void *command_queue,
                            overlay::Resolution resolution,
                            const StreamSpace &space) const = 0;
        
        virtual Texture get(const void *command_queue,
                            overlay::Resolution resolution) const;
        
        virtual Texture get(const void *command_queue) const;
    
        explicit image_cache(size_t cache_index);

    protected:
        
        
        Texture get(const void *command_queue,
                    overlay::Resolution resolution,
                    const std::string &file,
                    const StreamSpace &space = StreamSpace()) const ;
        
        Texture get(const void *command_queue,
                    overlay::Resolution resolution,
                    const uint8_t *image_buffer,
                    size_t length,
                    const StreamSpace &space = StreamSpace()) const;
        
        Texture get(const void *command_queue,
                    overlay::Resolution resolution,
                    const std::vector<uint8_t>& buffer,
                    const StreamSpace &space = StreamSpace()) const;
        
        ~image_cache();
    
    private:
        mutable OverlayImages cache_;
        mutable std::mutex mutex_;
        size_t cache_index_;
    };
    
    /***
     * Common singleton interface
     * @tparam T
     */
    template<typename T, size_t N>
    class SimpleImageCache {
    public:
        
        static T &Instance() {
          static T instance(N);
          return instance;
        }
    
    protected:
        SimpleImageCache() = default;
        ~SimpleImageCache() = default;
    
    public:
        SimpleImageCache(SimpleImageCache const &) = delete;
        SimpleImageCache &operator=(SimpleImageCache const &) = delete;
    };
    
    template<typename T, size_t N>
    class ControlledImageCache {
    public:
        
        using InstanceType = T;
        
        static InstanceType& Instance() {
          static InstanceType* instance = nullptr;
          static std::once_flag flag;
          std::call_once(flag, [&]{
              if (!instance)
              {
                instance = new InstanceType(N);
              }
          });
          return *instance;
        }
        
        static
        void CreateInstance() {
          InstanceType& p = Instance();
        }
        
        static
        void DestroyInstance() {
          InstanceType& p = Instance();
          delete &p;
        }
    
    protected:
        ControlledImageCache() = default;
        ~ControlledImageCache() = default;
    
    public:
        ControlledImageCache(ControlledImageCache const &) = delete;
        ControlledImageCache &operator=(ControlledImageCache const &) = delete;
    };
    
    #if defined(DEHANCER_CONTROLLED_SINGLETON)
    template<class T, size_t N>using ImageCache=ControlledImageCache<T,N>;
    #else
    template<class T, size_t N>using ImageCache=SimpleImageCache<T,N>;
    #endif
}