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
      
      auto kernel = dehancer::HistogramImage(command_queue);
      kernel.set_options({
        .ignore_edges = true,
        .edges = {
                .left_trim = 1.99f,
                .right_trim = 1.99f
        }
      });
      kernel.set_source(input_text.get_texture());
      kernel.process();
      
      const auto& histogram = kernel.get_histogram();
  
      using ch = dehancer::math::Channel::Index;
      float clipping_low = 0.1f/100.0f;
      float clipping_high = 0.1f/100.0f;
  
      for(int i = 0; i < (int)histogram.get_size().size; i++){
        std::cout << "["<<i<<"] = "
                  << "  " << (unsigned long)histogram.get_channel(ch::red)[i]
                  << ", " << (unsigned long)histogram.get_channel(ch::green)[i]
                  << ", " << (unsigned long)histogram.get_channel(ch::blue)[i]
                  << " :: " << (unsigned long)histogram.get_channel(ch::luma)[i]
                  << std::endl;
      }
  
      std::cout << "  clipped lower luma: "<< (int)histogram.get_channel(ch::luma).lower(clipping_low)  << std::endl;
      std::cout << " clipped higher luma: "<< (int)(histogram.get_channel(ch::luma).higher(clipping_high) * (float )histogram.get_size().size) << std::endl;
  
      std::cout << "   clipped lower red: "<< (int)histogram.get_channel(ch::red).lower(clipping_low)  << std::endl;
      std::cout << "  clipped higher red: "<< (int)(histogram.get_channel(ch::red).higher(clipping_high) * (float )histogram.get_size().size) << std::endl;
      
    }
    catch (const std::runtime_error &e) {
      std::cerr << "Kernel error: " << e.what() << std::endl;
    }
    return 0;
};
