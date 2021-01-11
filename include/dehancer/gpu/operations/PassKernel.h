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
        PassKernel(const void *command_queue,
                   const Texture &source = nullptr,
                   const Texture &destination = nullptr,
                   bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                   const std::string &library_path = "");
    };
}