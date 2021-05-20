//
// Created by denn nevera on 2019-07-23.
//

#include "dehancer/gpu/clut/CLut3DIdentity.h"

namespace dehancer {
    
    CLut3DIdentity::CLut3DIdentity(
            const void *command_queue,
            uint lut_size,
            bool wait_until_completed,
            const std::string &library_path):
            Function(command_queue, "kernel_make3DLut", wait_until_completed, library_path),
            CLut(),
            lut_size_(lut_size)
    {
      
      texture_ = make_texture(lut_size,lut_size,lut_size);
      
      execute([this](CommandEncoder& compute_encoder) {
          compute_encoder.set(texture_,0);
          compute_encoder.set(float2({1,0}),1);
          return CommandEncoder::Size::From(texture_);
      });
      
    }
    
    CLut3DIdentity::~CLut3DIdentity() = default;
}