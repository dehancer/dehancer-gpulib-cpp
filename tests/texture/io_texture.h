//
// Created by denn on 31.12.2020.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

auto io_texture_test = [] (int dev_num,
                           const void* command_queue,
                           const std::string& platform,
                           const std::string& input_image,
                           const std::string& output_image,
                           int image_index) {

    std::cout << "Load file: " << input_image << std::endl;

    auto input_text = dehancer::TextureInput(command_queue);
    
    auto command = dehancer::Command(command_queue);
    
    std::cout << "Maximum texture 1D size: " << command.get_max_texture_size(dehancer::TextureDesc::Type::i1d) << std::endl;
    std::cout << "Maximum texture 2D size: " << command.get_max_texture_size(dehancer::TextureDesc::Type::i2d) << std::endl;
    std::cout << "Maximum texture 3D size: " << command.get_max_texture_size(dehancer::TextureDesc::Type::i3d) << std::endl;
    
    std::ifstream ifs(input_image, std::ios::binary);
    ifs >> input_text;
    
    auto texture = input_text.get_texture();
    auto native_texture = texture->get_memory();
    
    auto texture_from_native = dehancer::TextureHolder::Make(command_queue,native_texture);
    
    auto output_text = dehancer::TextureOutput(command_queue, texture_from_native, {
            .type = test::type,
            .compression = test::compression
    });

    {
      std::ofstream os(output_image, std::ostream::binary | std::ostream::trunc);
      if (os.is_open()) {
        os << output_text << std::flush;

        std::cout << "Save to: " << output_image << std::endl;

      } else {
        std::cerr << "File: " << output_image << " could not been opened..." << std::endl;
      }
    }

    return 0;
};

