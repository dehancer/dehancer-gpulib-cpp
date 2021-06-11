//
// Created by denn nevera on 2019-07-23.
//

#include "dehancer/gpu/clut/CLut2DIdentity.h"

namespace dehancer {

    CLut2DIdentity::CLut2DIdentity(
            const void *command_queue,
            size_t lut_size,
            bool wait_until_completed ,
            const std::string &library_path ):
            Function(command_queue, "kernel_make2DLut", wait_until_completed, library_path),
            CLut(),
            level_((size_t)sqrtf((float)lut_size)),
            lut_size_(lut_size)
    {

      auto dimension = level_*level_*level_;
      texture_ = make_texture(dimension,dimension);

      execute([this](CommandEncoder& compute_encoder) {
    
          compute_encoder.set(texture_, 0);
          compute_encoder.set(float2({1,0}),1);
          compute_encoder.set(uint(level_),2);

          return CommandEncoder::Size::From(texture_);
      });

    }

    CLut2DIdentity::~CLut2DIdentity() = default;
}