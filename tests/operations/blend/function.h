//
// Created by denn on 01.01.2021.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

struct test_blend_options {
    DCHR_BlendingMode mode;
    float opacity = 0.5;
    std::string mode_name;
};

static std::vector<test_blend_options> options = {
        {
                .mode = DCHR_Normal,
                .mode_name = "normal"
        },
        {
                .mode = DCHR_Color,
                .mode_name = "color"
        },
        {
                .mode = DCHR_Luminosity,
                .mode_name = "luminosity"
        },
        {
                .mode = DCHR_Overlay,
                .mode_name = "overlay"
        },
        {
                .mode = DCHR_Mix,
                .mode_name = "mix"
        },
        {
                .mode = DCHR_Min,
                .mode_name = "min"
        },
        {
                .mode = DCHR_Max,
                .mode_name = "max"
        },
        {
                .mode = DCHR_Add,
                .mode_name = "add"
        },
};

int function_test_blend (int dev_num,
                         const void* command_queue,
                         const std::string& platform,
                         const std::string& input_image,
                         const std::string& output_image,
                         int image_index,
                         const test_blend_options& opt
) {
  try {
    
    auto grid_kernel = dehancer::Function(command_queue,"kernel_grid");
    auto grid_text = grid_kernel.make_texture(800, 400);
    
    grid_kernel.execute([&grid_text](dehancer::CommandEncoder& command_encoder){
        int levels = 6;
        
        command_encoder.set(levels, 0);
        command_encoder.set(grid_text, 1);
        
        return dehancer::CommandEncoder::Size::From(grid_text);
    });
    
    
    std::cout << "Load file: " << input_image << std::endl;
    
    /***
     * Load image to texture
     */
    auto input_text = dehancer::TextureInput(command_queue);
    std::ifstream ifs(input_image, std::ios::binary);
    ifs >> input_text;
    
    auto output_text = dehancer::TextureOutput(command_queue,
                                               input_text.get_texture()->get_width(), input_text.get_texture()->get_height(),
                                               nullptr,
                                               {
                                                       .type = test::type,
                                                       .compression = test::compression
                                               });
    
    auto kernel = dehancer::BlendKernel(command_queue, 0.5);
    
    kernel.set_source(input_text.get_texture());
    kernel.set_destination(output_text.get_texture());
    kernel.set_overlay(grid_text);
    kernel.set_mode(opt.mode);
    //kernel.set_opacity(opt.opacity);
    
    kernel.process();
    
    {
      std::string outp = opt.mode_name; outp += "-" + output_image;
      std::ofstream os(outp, std::ostream::binary | std::ostream::trunc);
      if (os.is_open()) {
        os << output_text << std::flush;
        
        std::cout << "Save to: " << outp << std::endl;
        
      } else {
        std::cerr << "File: " << outp << " could not been opened..." << std::endl;
      }
    }
    
  }
  catch (const std::runtime_error &e) {
    std::cerr << "Kernel error: " << e.what() << std::endl;
  }
  return 0;
};

auto function_test =  [] (int dev_num,
                          const void* command_queue,
                          const std::string& platform,
                          const std::string& input_image,
                          const std::string& output_image,
                          int image_index) {
    
    for (auto v: options) {
      function_test_blend(dev_num, command_queue, platform, input_image, output_image, image_index, v);
    }
    
    return 0;
};
