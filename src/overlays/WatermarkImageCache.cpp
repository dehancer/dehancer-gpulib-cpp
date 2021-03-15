//
// Created by denn on 15.03.2021.
//

#include "dehancer/gpu/overlays/WatermarkImageCache.h"


extern unsigned char dehancer_watermark_1K[];
extern unsigned int dehancer_watermark_1K_len;

extern unsigned char dehancer_watermark_4K[];
extern unsigned int dehancer_watermark_4K_len;

extern unsigned char dehancer_watermark_8K[];
extern unsigned int dehancer_watermark_8K_len;

namespace dehancer::overlay {
    
    static std::vector<ItemInfo> watermarks = {
            {
                    .resolution = Resolution::R1K,
                    .name = "watermark_1k",
                    .buffer = (uint8_t*)dehancer_watermark_1K,
                    .length = (size_t) dehancer_watermark_1K_len
            },
            {
                    .resolution = Resolution::R4K,
                    .name = "watermark_4k",
                    .buffer = (uint8_t*)dehancer_watermark_4K,
                    .length = (size_t) dehancer_watermark_4K_len,
            },
            {
                    .resolution = Resolution::R8K,
                    .name = "watermark_8k",
                    .buffer = (uint8_t*)dehancer_watermark_8K,
                    .length = (size_t) dehancer_watermark_8K_len,
            }
    };
    
    
    Texture
    watermark_image::get (const void *command_queue, overlay::Resolution resolution, const StreamSpace &space) const {
      auto t = image_cache::get(command_queue, resolution, nullptr, 0, space);
      if (!t) {
        for (auto& w: watermarks) {
          if (w.resolution==resolution){
            t = image_cache::get(command_queue, resolution, w.buffer, w.length, space);
          }
        }
        assert(t);
      }
      return t;
    }
    
    const std::vector<ItemInfo> &watermark_image::available () const {
      return watermarks;
    }
    
    Texture watermark_image::get (const void *command_queue, overlay::Resolution resolution) const {
      return image_cache::get(command_queue, resolution);
    }
}
