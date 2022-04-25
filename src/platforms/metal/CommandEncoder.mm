//
// Created by denn nevera on 10/11/2020.
//

#include "CommandEncoder.h"
#import <Metal/Metal.h>

namespace dehancer::metal {
    void CommandEncoder::set(const Texture &texture, int index)  {
      if (command_encoder_) {
        auto ce = static_cast<id<MTLComputeCommandEncoder>>( (__bridge id) (void*)command_encoder_);
        id<MTLTexture> t = static_cast<id <MTLTexture>>((__bridge id)texture->get_memory());
        [ce setTexture:t atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass texture to null kernel");
      }
    }

    void CommandEncoder::set(const void *bytes, size_t bytes_length, int index)  {
      if (command_encoder_){
        auto ce = static_cast<id<MTLComputeCommandEncoder>>( (__bridge id) (void*)command_encoder_);
        [ce setBytes:bytes length:bytes_length atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass bytes to null kernel");
      }
    }

    void CommandEncoder::set(const Memory &memory, int index) {
      if (command_encoder_) {
        auto ce = static_cast<id<MTLComputeCommandEncoder>>( (__bridge id) (void*)command_encoder_);
        id<MTLBuffer> buffer = static_cast<id <MTLBuffer>>((__bridge id)memory->get_memory());
        [ce setBuffer:buffer offset:0 atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass buffer to null kernel");
      }
    }
    
    void CommandEncoder::set (const StreamSpace &p, int index) {
      StreamSpace copy = p;
      set(&copy, sizeof(copy), index);
    }
}