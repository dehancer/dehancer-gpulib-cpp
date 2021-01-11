//
// Created by denn nevera on 2019-08-02.
//

#pragma once

#include "dehancer/gpu/Kernel.h"

namespace dehancer {
    
    /**
     * Bypass kernel.
     */
    class PassKernel: public Kernel {
    
    public:
        explicit PassKernel(const void *command_queue,
                   const Texture &source,
                   const Texture &destination,
                   bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                   const std::string &library_path = "");
    
        explicit PassKernel(const void *command_queue,
                   bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                   const std::string &library_path = "");
    };
}