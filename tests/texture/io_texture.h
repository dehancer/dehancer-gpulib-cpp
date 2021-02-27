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

    std::ifstream ifs(input_image, std::ios::binary);
    ifs >> input_text;

    auto output_text = dehancer::TextureOutput(command_queue, input_text.get_texture(), {
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

