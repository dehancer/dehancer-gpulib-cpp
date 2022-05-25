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
                        StreamSpace space,
                        StreamSpaceDirection direction,
                        float impact=1.0f,
                        bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                        const std::string &library_path="");
        
        void setup(CommandEncoder &commandEncoder) override ;
        
        void set_space(const StreamSpace& space);
        void set_direction(StreamSpaceDirection direction);
        void set_impact(float impact);
        
        const StreamSpace& get_space() const;
        StreamSpaceDirection get_direction() const;
        float get_impact() const;
        
    private:
        StreamSpace space_{};
        StreamSpaceDirection direction_{};
        float impact_;
    };
}
