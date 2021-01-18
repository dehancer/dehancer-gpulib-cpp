//
// Created by denn on 18.01.2021.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/kernels/types.h"

namespace dehancer {
    
    /**
     * Bypass kernel.
     */
    class BlendKernel: public Kernel {
    
    public:
        explicit BlendKernel(const void *command_queue,
                             const Texture &source,
                             const Texture &destination,
                             const Texture &overlay,
                             float opacity = 1.0f,
                             DCHR_BlendingMode mode = DCHR_Normal,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
        
        explicit BlendKernel(const void *command_queue,
                             const Texture &overlay,
                             float opacity = 1.0f,
                             DCHR_BlendingMode mode = DCHR_Normal,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
    
        explicit BlendKernel(const void *command_queue,
                             float opacity = 1.0f,
                             DCHR_BlendingMode mode = DCHR_Normal,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
    
        void set_overlay(const Texture &overlay);
        void set_opacity(float opacity);
        void set_mode(DCHR_BlendingMode mode);
        
        void setup(CommandEncoder &encoder) override;
        
    private:
        float opacity_;
        DCHR_BlendingMode mode_;
        Texture overlay_;
    };
}