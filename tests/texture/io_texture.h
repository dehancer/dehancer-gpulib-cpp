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

  try {
    std::cout << "Load file: " << input_image << std::endl;
  
    auto pixel_format = dehancer::TextureDesc::PixelFormat::rgba32float;
    
    auto input_text = dehancer::TextureInput(command_queue, pixel_format);
    
    auto command = dehancer::Command(command_queue);
  
    auto texture_info_1d = command.get_texture_info(dehancer::TextureDesc::Type::i1d);
    auto texture_info_2d = command.get_texture_info(dehancer::TextureDesc::Type::i2d);
    auto texture_info_3d = command.get_texture_info(dehancer::TextureDesc::Type::i3d);
  
    std::cout << "Maximum texture 1D size: " << texture_info_1d.max_width << std::endl;
    std::cout << "Maximum texture 2D size: " << texture_info_2d.max_width << "x" << texture_info_2d.max_height
              << std::endl;
    std::cout << "Maximum texture 3D size: " << texture_info_3d.max_width << "x" << texture_info_3d.max_height << "x"
              << texture_info_3d.max_depth << std::endl;
  
    std::ifstream ifs(input_image, std::ios::binary);
    //ifs >> input_text;
    std::vector<uint8_t> image_buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::vector<uint8_t> image_bytes;
    std::size_t width = 0, height = 0, channels = 0;
    
//    dehancer::TextureInput::image_to_data(image_buffer,
//                                          pixel_format,
//                                          image_bytes,
//                                          width,
//                                          height,
//                                          channels);
//
//    auto error = input_text.load_from_data(image_bytes, width, height);
    
    auto error = input_text.load_from_image(image_buffer);

    if (error) {
      std::cerr << " Image to data error: " << error << std::endl;
      return -1;
    }
  
    auto desc = input_text.get_texture()->get_desc();
    desc.pixel_format = dehancer::TextureDesc::PixelFormat::rgba16float;
  
    auto texture_16 = desc.make(command_queue);
  
  
    auto texture = input_text.get_texture();
  
    dehancer::PassKernel(command_queue, input_text.get_texture(), texture_16, true).process();
  
    auto native_texture = texture_16->get_memory();
  
    auto texture_from_native = dehancer::TextureHolder::Make(command_queue, native_texture);
  
//    texture_from_native = dehancer::TextureHolder::Rotate90(texture_from_native,
//                                                            dehancer::Rotate90Mode::up);
//
//    using mode = dehancer::FlipMode;
//    texture_from_native = dehancer::TextureHolder::Flip(texture_from_native,
//                                                        mode::horizontal|mode::vertical);

  
    auto cropped_texture =  texture_from_native; //dehancer::TextureHolder::Crop(texture_from_native, 0.1f, 0.0f, 0.0f, 0.2f);
  
    if (!cropped_texture) {
      std::cout << "Failed to crop texture: ..." << std::endl;
      return 1;
    }

    auto output_text = dehancer::TextureOutput(command_queue, cropped_texture, {
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
  
  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
    return 0;
};

