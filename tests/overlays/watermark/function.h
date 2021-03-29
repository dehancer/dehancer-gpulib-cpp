//
// Created by denn on 01.01.2021.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"


auto function_test =  [] (int dev_num,
                          const void* command_queue,
                          const std::string& platform,
                          const std::string& input_image,
                          const std::string& output_image,
                          int image_index) {
    
    try {
      std::cout << "Load file: " << input_image << std::endl;
      
      /***
       * Load image to texture
       */
      auto input_text = dehancer::TextureInput(command_queue);
      std::ifstream ifs(input_image, std::ios::binary);
      ifs >> input_text;
      
      using WCache = dehancer::overlay::WatermarkImageCache;
      
      auto resolution = dehancer::overlay::resolution_from(input_text.get_texture());
      
      std::cout << " ### Resolution for image: " << static_cast<int>(resolution) << std::endl;
      
      std::vector<dehancer::overlay::Resolution> rs = {
              dehancer::overlay::Resolution::LandscapeR1K,
              dehancer::overlay::Resolution::LandscapeR4K,
              dehancer::overlay::Resolution::LandscapeR8K
      };
      
      for (auto r: rs) {
        
        auto overlay = WCache::Instance().get(command_queue, r);
        
        auto output_text = dehancer::TextureOutput(command_queue, input_text.get_texture(), {
                .type = test::type,
                .compression = test::compression
        });
        
        auto kernel = dehancer::OverlayKernel(command_queue);
        
        kernel.set_interpolation(dehancer::ResampleKernel::Mode::bilinear);
        kernel.set_overlay(overlay);
        kernel.set_source(input_text.get_texture());
        kernel.set_destination(output_text.get_texture());
        
        kernel.process();
        
        {
          
          std::ostringstream osss; osss << static_cast<int>(r) << "-" << output_image;
          
          std::ofstream os(osss.str(), std::ostream::binary | std::ostream::trunc);
          
          if (os.is_open()) {
            os << output_text << std::flush;
            
            std::cout << "Save to: " << osss.str() << std::endl;
            
          } else {
            std::cerr << "File: " << osss.str() << " could not been opened..." << std::endl;
          }
        }
      }
    }
    catch (const std::runtime_error &e) {
      std::cerr << "Kernel error: " << e.what() << std::endl;
    }
    
    return 0;
};
