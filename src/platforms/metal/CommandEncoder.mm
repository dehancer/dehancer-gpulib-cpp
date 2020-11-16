//
// Created by denn nevera on 10/11/2020.
//

#include "CommandEncoder.h"

namespace dehancer::metal {
    void CommandEncoder::set(const Texture &texture, int index)  {
      if (command_encoder_) {
        id<MTLTexture> t = static_cast<id <MTLTexture>>((__bridge id)texture->get_memory());
        [command_encoder_ setTexture:t atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass texture to null kernel");
      }
    }

    void CommandEncoder::set(const void *bytes, size_t bytes_length, int index)  {
      if (command_encoder_){
        [command_encoder_ setBytes:bytes length:bytes_length atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass bytes to null kernel");
      }
    }
}