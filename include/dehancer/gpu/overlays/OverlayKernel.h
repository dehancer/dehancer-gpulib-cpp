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
        
        struct Options {
            float opacity;
            bool  horizontal_flipped;
            bool  vertical_flipped;
        };
        
        explicit OverlayKernel(const void *command_queue,
                               const Texture &source,
                               const Texture &destination,
                               const Texture &overlay,
                               Options options = Options({1.0f,false, false}),
                               ResampleKernel::Mode interpolation = ResampleKernel::Mode::bilinear,
                               bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                               const std::string &library_path = "");
        
        explicit OverlayKernel(const void *command_queue,
                               const Texture &overlay,
                               Options options = Options({1.0f,false, false}),
                               ResampleKernel::Mode interpolation = ResampleKernel::Mode::bilinear,
                               bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                               const std::string &library_path = "");
        
        explicit OverlayKernel(const void *command_queue,
                               Options options = Options({1.0f,false, false}),
                               ResampleKernel::Mode interpolation = ResampleKernel::Mode::bilinear,
                               bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                               const std::string &library_path = "");
        
        void set_overlay(const Texture &overlay);
        void set_options(Options options);
        void set_interpolation(ResampleKernel::Mode mode);
        void set_destination(const Texture &destination) override;
        void setup(CommandEncoder &encoder) override;
        
    private:
        Texture overlay_src_;
        Texture overlay_base_;
        ResampleKernel::Mode interpolation_mode_;
        Options options_;
        float2 overlay_offset_;
        
        void resize_overlay();
    };
}