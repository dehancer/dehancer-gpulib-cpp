//
// Created by denn nevera on 09/11/2020.
//

#include "Command.h"

namespace dehancer::metal {
    
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
      
      if (depth>1) {
        type = TextureDesc::Type::i3d;
      }
      else if (height==1) {
        type = TextureDesc::Type::i1d;
      }
      
      dehancer::TextureDesc desc = {
              .width = width,
              .height = height,
              .depth = depth,
              #if defined(IOS_SYSTEM)
              .pixel_format = TextureDesc::PixelFormat::rgba16float,
              #elif defined(DEHANCER_3DLUT_32FLOAT) || defined(DEHANCER_GPU_CUDA)
              .pixel_format = TextureDesc::PixelFormat::rgba32float,
              #else
              .pixel_format = TextureDesc::PixelFormat::rgba16float,
              #endif
              .type = type,
              .mem_flags = TextureDesc::MemFlags::read_write
      };
      
      void *q = reinterpret_cast<void*>(this->get_command_queue());
      
      return TextureHolder::Make(q, desc, nullptr);
    }
}
