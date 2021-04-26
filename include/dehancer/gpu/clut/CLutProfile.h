//
// Created by denn on 26.04.2021.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLutTransform.h"
#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/StreamSpace.h"

namespace dehancer {
    
    class CLutProfile: public CLut{
    
    public:
    
        enum class Format: int {
            square,
            /// TODO:
            //hald,
            //cube,
        };
    
        /**
         * Create profile lut from any source and convert to 3D
         * @param command_queue
         * @param source
         * @param format
         * @param space
         * @param direction
         * @param wait_until_completed
         * @param library_path
         */
        CLutProfile(const void *command_queue,
                    const std::vector<std::uint8_t>& source,
                    Format format,
                    const StreamSpace &space = StreamSpace::create_identity(),
                    StreamSpace::Direction direction = StreamSpace::Direction::none,
                    bool wait_until_completed = Function::WAIT_UNTIL_COMPLETED,
                    const std::string &library_path = "");

    private:
        std::unique_ptr<CLutTransform> transformer_;
    };
}
