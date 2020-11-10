//
// Created by denn nevera on 09/11/2020.
//

#include "OCLCommand.h"

namespace dehancer::opencl {

    Command::Command(const void *command_queue, bool wait_until_completed):
    OCLContext(command_queue),
    wait_until_completed_(wait_until_completed)
    {
    }

    Texture Command::make_texture(size_t width, size_t height, size_t depth) {
      ///
      /// TODO: right desc
      ///
      dehancer::TextureDesc desc = {
              .width = width,
              .height = height,
              .depth = depth,
              .pixel_format = TextureDesc::PixelFormat::rgba32float,
              .type = TextureDesc::Type::i2d
      };
      return TextureHolder::Make(get_command_queue(),desc);
    }
}
