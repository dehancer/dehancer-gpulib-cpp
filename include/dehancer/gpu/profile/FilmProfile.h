//
// Created by denn on 26.04.2021.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLutTransform.h"
#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/StreamSpace.h"
#include "dehancer/MLutXmp.h"

namespace dehancer {
    
    class FilmProfile: public Command {

    public:
        
        enum Type: int {
            under  = 0,
            normal = 1,
            over   = 2
        };
        
        explicit FilmProfile(
                const void *command_queue,
                const StreamSpace &space = StreamSpace::create_identity(),
                StreamSpace::Direction direction = StreamSpace::Direction::none,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED);
    
        Error load(const MLutXmp &xmp);
        
        const std::shared_ptr<CLut>& get(Type type) const;

    public:
        StreamSpace space_;
        StreamSpace::Direction direction_;
        std::array<std::shared_ptr<CLut>, sizeof(Type)> cluts_;
    };
}
