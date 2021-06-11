//
// Created by denn on 25.04.2021.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLut.h"

namespace dehancer {
    
    class CLutTransformFunction: public Function {
    public:
        CLutTransformFunction(
                const void *command_queue,
                const std::string &kernel_name,
                const Texture &source,
                const Texture &target,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                const std::string &library_path = "");
    };
}

