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
                const StreamSpace &space = stream_space_identity(),
                StreamSpaceDirection direction = StreamSpaceDirection::DHCR_None,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED);
    
        explicit FilmProfile(
                const void *command_queue,
                CLut::Type type,
                const StreamSpace &space = stream_space_identity(),
                StreamSpaceDirection direction = StreamSpaceDirection::DHCR_None,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED);
    
        Error load(const MLutXmp &xmp);
        
        const std::shared_ptr<CLut>& get(Type type) const;

    private:
        StreamSpace space_;
        StreamSpaceDirection direction_;
        std::array<std::shared_ptr<CLut>, sizeof(Type)> cluts_;
        CLut::Type type_;
    };
}
