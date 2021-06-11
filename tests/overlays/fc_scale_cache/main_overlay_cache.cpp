//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../../include/run_test.h"

#include <iostream>

auto io_texture_test = [] (int num,
                           const void* command_queue,
                           const std::string& platform) {
    
    using FC_Cache = dehancer::overlay::FalsecolorScaleImageCache;
    
    for (auto& v: FC_Cache::Instance().available()) {
      
      std::cout << "image["<<v.name<<"]: " << static_cast<int>(v.resolution) << " : " << v.length << std::endl;
      
      auto t = FC_Cache::Instance().get(command_queue, v.resolution);
      
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

TEST(TEST, CUDA_FALSE_COLOR_IMAGES_CACHE) {
  
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