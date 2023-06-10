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
  
      dehancer::DHCR_StreamSpace_TransformFunc transform_function = {
              .is_identity = static_cast<bool_t>(false),
              .cs_forward_matrix = dehancer::stream_matrix_transform_identity(),
              .cs_inverse_matrix = dehancer::stream_matrix_transform_identity(),
              .cs_params = {
                      .gamma = dehancer::ocio::REC709_22::gamma_parameters,
                      .log = {
                              .enabled =  static_cast<bool_t>(false)
                      }
              },
      };
  
      auto space = (dehancer::StreamSpace) {
              .type = dehancer::DHCR_ColorSpace,
              .expandable = false,
              .transform_func = transform_function,
              .id = "rec709",
              .name="Rec709",
      };
      
      /***
       * Load image to texture
       */
      auto input_text = dehancer::TextureInput(command_queue);
      std::ifstream ifs(input_image, std::ios::binary);
      ifs >> input_text;
      
      
      auto kernel = dehancer::HistogramImage(command_queue);
      
      kernel.set_options({
        .edges = {
                .ignore = false,
                .left_trim = 1.0f,
                .right_trim = 1.0f
        },
        .transform = {
          .enabled = true,
          .space = space,
          .direction = DHCR_Forward
        },
        .luma_type = dehancer::HistogramImage::LumaType::YCbCr
      });
      kernel.set_source(input_text.get_texture());
      kernel.process();
      
      const auto& histogram = kernel.get_histogram();
  
      using ch = dehancer::math::Channel::Index;
      float clipping_low = 0.1f/100.0f;
      float clipping_high = 0.1f/100.0f;
  
      auto r = histogram.get_channel(ch::red);
      auto g = histogram.get_channel(ch::green);
      auto b = histogram.get_channel(ch::blue);
      auto l = histogram.get_channel(ch::luma);
  
      for(int i = 0; i < (int)histogram.get_size().size; i++){
        if (r[i] == 0 && g[i] == 0 && b[i] == 0 && l[i] == 0) continue;
        std::cout << "["<<i<<"] = "
                  << "  "   << (unsigned long)r[i]
                  << ", "   << (unsigned long)g[i]
                  << ", "   << (unsigned long)b[i]
                  << " :: " << (unsigned long)l[i]
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
