//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../../include/run_test.h"

#include <iostream>

struct watermark_buff {
    std::string name;
    uint8_t* buffer;
    size_t   length;
};

extern unsigned char dehancer_watermark_1K[];
extern unsigned int dehancer_watermark_1K_len;

extern unsigned char dehancer_watermark_4K[];
extern unsigned int dehancer_watermark_4K_len;

extern unsigned char dehancer_watermark_8K[];
extern unsigned int dehancer_watermark_8K_len;

std::vector<watermark_buff> watermarks = {
        {
                .name = "watermark_1k",
                .buffer = (uint8_t*)dehancer_watermark_1K,
                .length = (size_t) dehancer_watermark_1K_len
        },
        {
                .name = "watermark_4k",
                .buffer = (uint8_t*)dehancer_watermark_4K,
                .length = (size_t) dehancer_watermark_4K_len
        },
        {
                .name = "watermark_8k",
                .buffer = (uint8_t*)dehancer_watermark_8K,
                .length = (size_t) dehancer_watermark_8K_len
        },
};

auto io_texture_test = [] (int num,
                           const void* command_queue,
                           const std::string& platform) {
    
    dehancer::overlay::image_cache cache(0);
    
    for (auto& v: watermarks) {
      std::cout << "image: " << v.length << std::endl;
      auto t = cache.get(command_queue, dehancer::overlay::Resolution::Default, v.buffer, v.length);
      
      auto output_text = dehancer::TextureOutput(command_queue, t, {
              .type = test::type,
              .compression = test::compression
      });
      
      {
        std::ostringstream os_name; os_name << v.name << test::ext;
        std::ofstream os(os_name.str(), std::ostream::binary | std::ostream::trunc);
        if (os.is_open()) {
          os << output_text << std::flush;
          
          std::cout << "Save to: " << os_name.str() << std::endl;
          
        } else {
          std::cerr << "File: " << os_name.str() << " could not been opened..." << std::endl;
        }
      }
      
    }
    
    return 0;
};

TEST(TEST, CUDA_OVERLAY_CACHE) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_on_devices("common", io_texture_test);
  
}

namespace dehancer::device {
    
    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
  
}