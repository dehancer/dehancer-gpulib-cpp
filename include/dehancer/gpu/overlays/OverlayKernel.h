//
// Created by denn on 18.01.2021.
//

#pragma once

#include "dehancer/gpu/operations/ResampleKernel.h"
#include "dehancer/gpu/kernels/types.h"

namespace dehancer {
    
    /**
     * Bypass kernel.
     */
    class OverlayKernel: public Kernel {
    
    public:
        using Kernel::Kernel;
      
        explicit OverlayKernel(const void *command_queue,
                             const Texture &source,
                             const Texture &destination,
                             const Texture &overlay,
                             float opacity = 1.0f,
                             ResampleKernel::Mode interpolation = ResampleKernel::Mode::bicubic,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
        
        explicit OverlayKernel(const void *command_queue,
                             const Texture &overlay,
                             float opacity = 1.0f,
                             ResampleKernel::Mode interpolation = ResampleKernel::Mode::bicubic,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
        
        explicit OverlayKernel(const void *command_queue,
                             float opacity = 1.0f,
                             ResampleKernel::Mode interpolation = ResampleKernel::Mode::bicubic,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
    
        void set_overlay(const Texture &overlay);
        void set_opacity(float opacity);
        void set_interpolation(ResampleKernel::Mode mode);
        
        void setup(CommandEncoder &encoder) override;
        
    private:
        Texture overlay_;
        float opacity_;
        ResampleKernel::Mode interpolation_mode_;
        bool vertical_flipped_;
    };
}