//
// Created by denn on 01.01.2021.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

static dehancer::ChannelsDesc::Transform transform_channels = {
        .type    = dehancer::ChannelsDesc::TransformType::log_linear,
        .slope   = {6.5f, 4.5f, 2.5f,   0},
        .offset  = {1.0f, 1.0f, 1.0f,   0},
        .enabled = {true,false,false,false},
        .direction = dehancer::ChannelsDesc::TransformDirection::forward,
//        .flags ={
//                .in_enabled = false,
//                .out_enabled = false
//        }
};

static std::array<float,4> transform_radiuses = {20.0f,00.0f,00.0f,0.0f};
//static std::array<float,4> transform_radiuses = {20.0f,20.0f,20.0f,0.0f};

static void run_kernel(int dev_num,
                       const void* command_queue,
                       const std::string& platform,
                       const std::string& input_image,
                       const std::string& output_image,
                       int image_index,
                       std::function<std::string (const dehancer::Texture& input, const dehancer::Texture& output)> block
                       ) {
  try {
    std::cout << "Load file: " << input_image << std::endl;
    
    /***
     * Load image to texture
     */
    auto input_text = dehancer::TextureInput(command_queue);
    std::ifstream ifs(input_image, std::ios::binary);
    ifs >> input_text;
    
    auto output_text = dehancer::TextureOutput(command_queue, input_text.get_texture(), {
            .type = test::type,
            .compression = test::compression,
    });
    
    auto func_type=block(input_text.get_texture(), output_text.get_texture());
    
    {
      std::ofstream os(func_type+"-"+output_image, std::ostream::binary | std::ostream::trunc);
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
}

auto gaussian_test =  [] (int dev_num,
                          const void* command_queue,
                          const std::string& platform,
                          const std::string& input_image,
                          const std::string& output_image,
                          int image_index) {
    
    run_kernel(dev_num,command_queue,platform,input_image,output_image,image_index,
               [command_queue](const dehancer::Texture& input, const dehancer::Texture& output){
       
                   auto kernel = dehancer::GaussianBlur(command_queue);
    
                   kernel.set_source(input);
                   kernel.set_destination(output);
                   kernel.set_accuracy(0.000001);
                   kernel.set_transform(transform_channels);
                   kernel.set_radius(transform_radiuses);
                   kernel.set_edge_mode(DHCR_ADDRESS_WRAP);
    
                   kernel.process();
                   
                   return "gaussian";
    });
    return 0;
};

auto box_test =  [] (int dev_num,
                          const void* command_queue,
                          const std::string& platform,
                          const std::string& input_image,
                          const std::string& output_image,
                          int image_index) {
  
    run_kernel(dev_num,command_queue,platform,input_image,output_image,image_index,
               [command_queue](const dehancer::Texture& input, const dehancer::Texture& output){
                   auto kernel = dehancer::BoxBlur(command_queue);
        
                   kernel.set_source(input);
                   kernel.set_destination(output);
                   kernel.set_radius(20);
                   kernel.set_edge_mode(DHCR_ADDRESS_WRAP);
        
                   kernel.process();
                   
                   return "box_blur";
               });
    return 0;
};
