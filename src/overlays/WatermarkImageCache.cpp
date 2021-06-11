//
// Created by denn on 15.03.2021.
//

#include "dehancer/gpu/overlays/WatermarkImageCache.h"

extern "C" unsigned char dehancer_watermark_16x9_1K[];
extern "C" unsigned int dehancer_watermark_16x9_1K_len;

extern "C" unsigned char dehancer_watermark_16x9_4K[];
extern "C" unsigned int dehancer_watermark_16x9_4K_len;

extern "C" unsigned char dehancer_watermark_16x9_8K[];
extern "C" unsigned int dehancer_watermark_16x9_8K_len;

namespace dehancer::overlay {
    
    static std::vector<ItemInfo> watermarks = {
            {
                    .resolution = Resolution::LandscapeR1K,
                    .name = "watermark_16x9_1k",
                    .buffer = (uint8_t*)dehancer_watermark_16x9_1K,
                    .length = (size_t) dehancer_watermark_16x9_1K_len
            },
            {
                    .resolution = Resolution::LandscapeR4K,
                    .name = "watermark_16x9_4k",
                    .buffer = (uint8_t*)dehancer_watermark_16x9_4K,
                    .length = (size_t) dehancer_watermark_16x9_4K_len,
            },
            {
                    .resolution = Resolution::LandscapeR8K,
                    .name = "watermark_16x9_8k",
                    .buffer = (uint8_t*)dehancer_watermark_16x9_8K,
                    .length = (size_t) dehancer_watermark_16x9_8K_len,
            }
    };
    
    
    Texture
    watermark_image::get (const void *command_queue, overlay::Resolution resolution, const StreamSpace &space) const {
      auto t = image_cache::get(command_queue, resolution, nullptr, 0, space);
      if (!t) {
        for (auto& w: watermarks) {
          if (w.resolution==resolution){
            t = image_cache::get(command_queue, resolution, w.buffer, w.length, space);
            break;
          }
        }
      }
      assert(t);
      return t;
    }
    
    const std::vector<ItemInfo> &watermark_image::available () const {
      return watermarks;
    }
    
    Texture watermark_image::get (const void *command_queue, overlay::Resolution resolution) const {
      return image_cache::get(command_queue, resolution);
    }
}
