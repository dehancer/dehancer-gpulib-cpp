//
// Created by denn on 26.04.2021.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLutTransform.h"
#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/StreamSpace.h"
#include "dehancer/CameraLutXmp.h"

namespace dehancer {
    
    class CameraProfile: public Command {
    
    public:
        
        explicit CameraProfile(
                const void *command_queue,
                const StreamSpace &space = StreamSpace::create_identity(),
                StreamSpace::Direction direction = StreamSpace::Direction::none,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED);
        
        Error load(const CameraLutXmp &xmp);
        
        const std::shared_ptr<CLut>& get() const;
    
    public:
        StreamSpace space_;
        StreamSpace::Direction direction_;
        std::shared_ptr<CLut> clut_;
    };
}