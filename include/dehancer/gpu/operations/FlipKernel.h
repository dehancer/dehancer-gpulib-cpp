//
// Created by denn nevera on 2019-08-02.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/kernels/types.h"

namespace dehancer {
    
    /**
     * Resample kernel
     */
    class FlipKernel: public Kernel {
    
    public:
        using Kernel::Kernel;
    
        /***
         * Flip mode
         */
        using Mode = FlipMode;
        
        explicit FlipKernel(const void *command_queue,
                                const Texture &source,
                                const Texture &destination,
                                Mode mode = Mode::nope,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");
        
        explicit FlipKernel(const void *command_queue,
                                Mode mode = Mode::nope,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");
    
        void setup(CommandEncoder &encoder) override;
    
        [[maybe_unused]] void set_mode(Mode mode);
        
    private:
        Mode mode_;
    };
}