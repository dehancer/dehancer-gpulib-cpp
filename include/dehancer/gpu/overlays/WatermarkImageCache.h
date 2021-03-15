//
// Created by denn nevera on 2019-08-03.
//

#pragma once

#include "dehancer/gpu/overlays/OverlayImageCache.h"

namespace dehancer::overlay {
    
    class watermark_image: public image_cache {
    public:
    
        using image_cache::image_cache;
        
        Texture get(const void *command_queue,
                    overlay::Resolution resolution,
                    const StreamSpace& space) const override ;
        
        Texture get(const void *command_queue, overlay::Resolution resolution) const override;
        
        const std::vector<ItemInfo> & available() const override;
        
    };
    
    class WatermarkImageCache: public ImageCache<watermark_image,1>{
    public:
        WatermarkImageCache() = default;
    };
}