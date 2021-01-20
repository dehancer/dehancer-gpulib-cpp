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
    class BlendKernel: public Kernel {
    
    public:
    
        enum Mode {
            normal = DCHR_Normal,
            luminosity = DCHR_Luminosity,
            color = DCHR_Color,
            mix = DCHR_Mix,
            overlay = DCHR_Overlay,
            min = DCHR_Min,
            max = DCHR_Max,
            add = DCHR_Add
        };
        
        explicit BlendKernel(const void *command_queue,
                             const Texture &source,
                             const Texture &destination,
                             const Texture &overlay,
                             float opacity = 1.0f,
                             Mode mode = normal,
                             ResampleKernel::Mode interpolation = ResampleKernel::Mode::bilinear,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
        
        explicit BlendKernel(const void *command_queue,
                             const Texture &overlay,
                             float opacity = 1.0f,
                             Mode mode = normal,
                             ResampleKernel::Mode interpolation = ResampleKernel::Mode::bilinear,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
    
        explicit BlendKernel(const void *command_queue,
                             float opacity = 1.0f,
                             Mode mode = normal,
                             ResampleKernel::Mode interpolation = ResampleKernel::Mode::bilinear,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
    
        void set_overlay(const Texture &overlay);
        void set_opacity(float opacity);
        void set_mode(Mode mode);
        void set_interpolation(ResampleKernel::Mode mode);
        
        void setup(CommandEncoder &encoder) override;
        
    private:
        float opacity_;
        Mode mode_;
        ResampleKernel::Mode interpolation_mode_;
        Texture overlay_;
    };
}