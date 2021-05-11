//
// Created by denn on 08.05.2021.
//

#include "dehancer/gpu/spaces/StreamTransform.h"
#include "dehancer/gpu/spaces/StreamSpaceCache.h"

namespace dehancer {
    
    StreamTransform::StreamTransform (const void *command_queue,
                                      const Texture &source,
                                      const Texture &destination,
                                      const StreamSpace &space,
                                      StreamSpaceDirection direction,
                                      float impact,
                                      bool wait_until_completed,
                                      const std::string &library_path):
            Kernel(command_queue, "kernel_stream_transform", source, destination, wait_until_completed, library_path),
            space_(space),
            direction_(direction),
            impact_(impact)
    {
    }
    
    void StreamTransform::setup (CommandEncoder &encoder) {
      
      auto transform_lut =
              StreamSpaceCache::Instance()
                      .get_lut(get_command_queue(), space_, direction_);
  
      bool transform_lut_enabled =
              !!transform_lut
              &&
              !space_.transform_lut.is_identity
              &&
              direction_ == DHCR_Forward
              ? space_.transform_lut.forward.enabled
              : space_.transform_lut.inverse.enabled;
  
      bool transform_function_enabled = !space_.transform_func.is_identity;
  
      if (transform_lut)
        encoder.set(transform_lut->get_texture(), 2);
      
      encoder.set(&space_,sizeof(space_),3);
      encoder.set(direction_,4);
      encoder.set(transform_lut_enabled,5);
      encoder.set(transform_function_enabled,6);
      encoder.set(impact_,7);
    }
}