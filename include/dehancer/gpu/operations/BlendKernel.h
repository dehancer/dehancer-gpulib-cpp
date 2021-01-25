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
        using Kernel::Kernel;
      
        enum Mode {
            normal = DHCR_Normal,
            luminosity = DHCR_Luminosity,
            color = DHCR_Color,
            mix = DHCR_Mix,
            overlay = DHCR_Overlay,
            min = DHCR_Min,
            max = DHCR_Max,
            add = DHCR_Add,
            subtract = DHCR_Subtract
        };
        
        explicit BlendKernel(const void *command_queue,
                             const Texture &source,
                             const Texture &destination,
                             const Texture &overlay,
                             const Texture &mask = nullptr,
                             float opacity = 1.0f,
                             Mode mode = normal,
                             ResampleKernel::Mode interpolation = ResampleKernel::Mode::bilinear,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
        
        explicit BlendKernel(const void *command_queue,
                             const Texture &overlay,
                             const Texture &mask,
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
        void set_mask(const Texture &mask);
        void set_opacity(float opacity);
        void set_mode(Mode mode);
        void set_interpolation(ResampleKernel::Mode mode);
        
        void setup(CommandEncoder &encoder) override;
        
    private:
        Texture overlay_;
        Texture mask_;
        float opacity_;
        Mode mode_;
        ResampleKernel::Mode interpolation_mode_;
        bool has_mask_;
    };
}