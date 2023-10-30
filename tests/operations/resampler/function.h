//
// Created by denn on 01.01.2021.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

#define RESIZE_ON 0

float scale = 1.0f/2.0f;

#if RESIZE_ON
auto  interpolation = dehancer::ResizeKernel::Mode::lanczos;
#else
auto  interpolation = dehancer::ResampleKernel::Mode::bilinear;
#endif

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
  
      size_t w = (size_t )((float)input_text.get_texture()->get_width() * scale);
      size_t h = (size_t )((float)input_text.get_texture()->get_height() * scale);
      
      auto output_text = dehancer::TextureOutput(command_queue,
                                                 w,h,
                                                 {
                                                         .type = test::type,
                                                         .compression = test::compression
                                                 });
      
#if RESIZE_ON
      auto resampler = dehancer::ResizeKernel(command_queue, interpolation, 0.5f);
#else
      auto resampler = dehancer::ResampleKernel(command_queue, interpolation);
#endif

      resampler.set_source(input_text.get_texture());
      resampler.set_destination(output_text.get_texture());

      resampler.process();
      
      {
        std::ofstream os(output_image, std::ostream::binary | std::ostream::trunc);
        if (os.is_open()) {
          os << output_text << std::flush;
          
          std::cout << "Save to: " << output_image << std::endl;
          
        } else {
          std::cerr << "File: " << output_image << " could not been opened..." << std::endl;
        }
      }
      
    }
    catch (const std::runtime_error &e) {
      std::cerr << "Kernel error: " << e.what() << std::endl;
    }
    return 0;
};
