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
       * Create 3D lut transformation
       */
      auto kernel_make3DLut_transform = dehancer::Function(command_queue, "kernel_make3DLut_transform");

      /***
       * Make empty 3D Lut
       */
      auto clut = dehancer::TextureDesc{
              .width = 64,
              .height= 64,
              .depth = 64,
              .type  = dehancer::TextureDesc::Type::i3d
      }.make(command_queue);

      /***
       * Compute transformation
       */
      kernel_make3DLut_transform.execute([&clut](dehancer::CommandEncoder &command_encoder) {
          command_encoder.set(clut, 0);
          command_encoder.set((dehancer::math::float2) {1, 0}, 1);
          return dehancer::CommandEncoder::Size::From(clut);
      });

      /***
      * Create 1D lut transformation
      */
      auto kernel_make1DLut_transform = dehancer::Function(command_queue, "kernel_make1DLut_transform");

      /***
       * Make empty curve
       */
      auto clut_curve = dehancer::TextureDesc{
              .width = 256,
              .height= 1,
              .depth = 1,
              .type  = dehancer::TextureDesc::Type::i1d
      }.make(command_queue);

      kernel_make1DLut_transform.execute([&clut_curve](dehancer::CommandEncoder &command_encoder) {
          command_encoder.set(clut_curve, 0);
          command_encoder.set((dehancer::math::float2) {0.5, 0.5}, 1);
          return dehancer::CommandEncoder::Size::From(clut_curve);
      });

      /***
       * Load image to texture
       */
      auto input_text = dehancer::TextureInput(command_queue);
      std::ifstream ifs(input_image, std::ios::binary);
      ifs >> input_text;

      auto output_text = dehancer::TextureOutput(command_queue, input_text.get_texture(), {
              .type = test::type,
              .compression = test::compression
      });

      /***
       * Execute transformation
       */
      auto kernel_transform = dehancer::Function(command_queue, "kernel_test_transform", true);

      kernel_transform.execute(
              [&input_text, &output_text, &clut, &clut_curve](dehancer::CommandEncoder &command_encoder) {
                  command_encoder.set(input_text.get_texture(), 0);
                  command_encoder.set(output_text.get_texture(), 1);
                  command_encoder.set(clut, 2);
                  command_encoder.set(clut_curve, 3);
                  return dehancer::CommandEncoder::Size::From(output_text.get_texture());
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
      std::cerr << "Kernel error: " << e.what() << std::endl;
    }
    return 0;
};
