//
// Created by denn nevera on 2019-08-02.
//

#pragma once

#include "dehancer/gpu/Kernel.h"

namespace dehancer {
    
    /**
     * Resample kernel
     */
    class ResampleKernel: public Kernel {
    
    public:
        
        /***
         * Resample mode
         */
        enum Mode {
            /***
             * Current version has linear mode
             */
            linear
        };
        
        explicit ResampleKernel(const void *command_queue,
                                const Texture &source,
                                const Texture &destination,
                                Mode mode = linear,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");
        
        explicit ResampleKernel(const void *command_queue,
                                Mode mode = linear,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");
    
        [[maybe_unused]] void set_mode(float Mode);
    };
}