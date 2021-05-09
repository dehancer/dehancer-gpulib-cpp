//
// Created by denn on 26.04.2021.
//

#include "dehancer/gpu/clut/CLutProfile.h"

namespace dehancer {
    
    CLutProfile::CLutProfile (const void *command_queue,
                              const std::vector<std::uint8_t> &source,
                              Format format,
                              const StreamSpace &space,
                              StreamSpaceDirection direction,
                              bool wait_until_completed,
                              const std::string &library_path):
            CLut()
    {
    
    }
}