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
  
      auto desc = input_text.get_texture()->get_desc();
      desc.pixel_format = dehancer::TextureDesc::PixelFormat::rgba16float;
      auto tmp_text = desc.make(command_queue);
      
      auto output_text = dehancer::TextureOutput(command_queue,
                                                 //tmp_text,
                                                 input_text.get_texture()->get_desc().width,
                                                 input_text.get_texture()->get_desc().height,
                                                 {
                                                         .type = test::type,
                                                         .compression = test::compression
                                                 });
      
      
      dehancer::ResampleKernel(command_queue, input_text.get_texture(), tmp_text, dehancer::ResampleKernel::Mode::bicubic, true).process();
//      dehancer::PassKernel(command_queue, input_text.get_texture(), tmp_text, true).process();
      
      auto kernel = dehancer::DilateKernel(command_queue, 4, 1, true);
//      auto kernel = dehancer::PassKernel(command_queue);
      
//      kernel.set_source(input_text.get_texture());
//      kernel.set_destination(tmp_text);
//      kernel.process();
  
      kernel.set_source(tmp_text);
      kernel.set_destination(output_text.get_texture());
      kernel.process();
  
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
