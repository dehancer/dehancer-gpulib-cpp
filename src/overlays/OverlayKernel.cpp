//
// Created by denn on 18.01.2021.
//

#include "dehancer/gpu/overlays/OverlayKernel.h"

namespace dehancer {
    
    OverlayKernel::OverlayKernel (const void *command_queue,
                              const Texture &base,
                              const Texture &destination,
                              const Texture &overlay,
                              float opacity,
                              ResampleKernel::Mode interpolation,
                              bool wait_until_completed,
                              const std::string &library_path):
            Kernel(command_queue, "kernel_overlay_image", base, destination, wait_until_completed, library_path),
            overlay_(overlay),
            opacity_(opacity),
            interpolation_mode_(interpolation),
            vertical_flipped_(true)
    {
    }
    
    OverlayKernel::OverlayKernel (const void *command_queue,
                              const Texture &overlay,
                              float opacity,
                              ResampleKernel::Mode interpolation,
                              bool wait_until_completed,
                              const std::string &library_path):
            OverlayKernel(command_queue, nullptr, nullptr,  overlay, opacity, interpolation, wait_until_completed, library_path)
    {
    }
    
    OverlayKernel::OverlayKernel (const void *command_queue,
                              float opacity,
                              ResampleKernel::Mode interpolation,
                              bool wait_until_completed,
                              const std::string &library_path):
            OverlayKernel(command_queue, nullptr, nullptr, nullptr, opacity, interpolation, wait_until_completed, library_path)
    {
    }
    
    void OverlayKernel::set_opacity (float opacity) {
      opacity_ = opacity;
    }
    
    void OverlayKernel::set_overlay (const Texture &overlay) {
      overlay_ = overlay;
    }
    
    void OverlayKernel::setup (CommandEncoder &encoder) {
      if (!overlay_) return;
      encoder.set(overlay_,2);
      encoder.set(opacity_,3);
      encoder.set(interpolation_mode_,4);
      encoder.set(vertical_flipped_,5);
    }
    
    void OverlayKernel::set_interpolation (ResampleKernel::Mode mode) {
      interpolation_mode_ = mode;
    }
    
}