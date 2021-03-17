//
// Created by denn on 15.03.2021.
//

#include "dehancer/gpu/overlays/FalsecolorScaleImageCache.h"


extern unsigned char dehancer_falsecolor_scale_1K[];
extern unsigned int dehancer_falsecolor_scale_1K_len;

extern unsigned char dehancer_falsecolor_scale_4K[];
extern unsigned int dehancer_falsecolor_scale_4K_len;

extern unsigned char dehancer_falsecolor_scale_8K[];
extern unsigned int dehancer_falsecolor_scale_8K_len;

namespace dehancer::overlay {
    
    static std::vector<ItemInfo> false_color_scales = {
            {
                    .resolution = Resolution::LandscapeR1K,
                    .name = "falsecolor_scale_1k",
                    .buffer = (uint8_t*)dehancer_falsecolor_scale_1K,
                    .length = (size_t) dehancer_falsecolor_scale_1K_len
            },
            {
                    .resolution = Resolution::LandscapeR4K,
                    .name = "falsecolor_scale_4k",
                    .buffer = (uint8_t*)dehancer_falsecolor_scale_4K,
                    .length = (size_t) dehancer_falsecolor_scale_4K_len,
            },
            {
                    .resolution = Resolution::LandscapeR8K,
                    .name = "falsecolor_scale_8k",
                    .buffer = (uint8_t*)dehancer_falsecolor_scale_8K,
                    .length = (size_t) dehancer_falsecolor_scale_8K_len,
            }
    };
    
    
    Texture
    false_color_scale_image::get (const void *command_queue, overlay::Resolution resolution, const StreamSpace &space) const {
      auto t = image_cache::get(command_queue, resolution, nullptr, 0, space);
      if (!t) {
        for (auto& w: false_color_scales) {
          if (w.resolution==resolution){
            t = image_cache::get(command_queue, resolution, w.buffer, w.length, space);
            break;
          }
        }
      }
      assert(t);
      return t;
    }
    
    const std::vector<ItemInfo> &false_color_scale_image::available () const {
      return false_color_scales;
    }
    
    Texture false_color_scale_image::get (const void *command_queue, overlay::Resolution resolution) const {
      return image_cache::get(command_queue, resolution);
    }
}
