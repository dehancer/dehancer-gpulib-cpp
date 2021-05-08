//
// Created by denn on 08.05.2021.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/StreamSpace.h"

namespace dehancer {
    class StreamTransform: public Kernel {
    public:
        StreamTransform(const void *command_queue,
                        const Texture &source,
                        const Texture &destination,
                        const StreamSpace &space,
                        StreamSpace::Direction direction,
                        float impact=1.0f,
                        bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                        const std::string &library_path="");
        
        void setup(CommandEncoder &commandEncoder) override ;
    
    private:
        StreamSpace space_;
        StreamSpace::Direction direction_;
        float impact_;
    };
}
