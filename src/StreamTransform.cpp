//
// Created by denn on 08.05.2021.
//

#include "dehancer/gpu/StreamTransform.h"

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
    
    void StreamTransform::setup (CommandEncoder &commandEncoder) {
      Kernel::setup(commandEncoder);
    }
}