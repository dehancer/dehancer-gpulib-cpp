//
// Created by denn nevera on 09/11/2020.
//

#include "Command.h"
#include "dehancer/Log.h"

namespace dehancer::cuda {

    Command::Command(const void *command_queue, bool wait_until_completed):
            Context(command_queue),
            wait_until_completed_(wait_until_completed)
    {
    }

    Texture Command::make_texture(size_t width, size_t height, size_t depth) {
      ///
      /// TODO: right desc
      ///

      TextureDesc::Type type = TextureDesc::Type::i2d;
      TextureDesc::PixelFormat pixel_format =  dehancer::Command::pixel_format_2d;
  
      if (depth>1) {
        type = TextureDesc::Type::i3d;
        pixel_format =  dehancer::Command::pixel_format_3d;
      }
      else if (height==1) {
        type = TextureDesc::Type::i1d;
        pixel_format =  dehancer::Command::pixel_format_1d;
      }

      dehancer::TextureDesc desc = {
              .width = width,
              .height = height,
              .depth = depth,
              .pixel_format = pixel_format,
              .type = type,
              .mem_flags = TextureDesc::MemFlags::read_write
      };
      
      //printf("Command::make_texture %zux%zux%zu pixel_format = %i, type = %i, 3d_pixel_format = %i\n", desc.width, desc.height, desc.depth, desc.pixel_format, desc.type, pixel_format);
      
      return TextureHolder::Make(get_command_queue(), desc, nullptr);
    }
}
