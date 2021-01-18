//
// Created by denn on 18.01.2021.
//

#include "dehancer/gpu/operations/BlendKernel.h"

namespace dehancer {
    
    BlendKernel::BlendKernel (const void *command_queue,
                              const Texture &base,
                              const Texture &destination,
                              const Texture &overlay,
                              float opacity,
                              DCHR_BlendingMode mode,
                              bool wait_until_completed,
                              const std::string &library_path):
            Kernel(command_queue, "kernel_blend", base, destination, wait_until_completed, library_path)
    {
      
    }
    
    BlendKernel::BlendKernel (const void *command_queue,
                              const Texture &overlay,
                              float opacity,
                              DCHR_BlendingMode mode,
                              bool wait_until_completed,
                              const std::string &library_path):
            BlendKernel(command_queue, nullptr, nullptr,  overlay, opacity, mode, wait_until_completed, library_path)
    {
      
    }
    
    BlendKernel::BlendKernel (const void *command_queue,
                              float opacity,
                              DCHR_BlendingMode mode,
                              bool wait_until_completed,
                              const std::string &library_path):
            BlendKernel(command_queue, nullptr, nullptr, nullptr, opacity, mode, wait_until_completed, library_path)
    {
    
    }
    
    void BlendKernel::set_opacity (float opacity) {
      opacity_ = opacity;
    }
    
    void BlendKernel::set_mode (DCHR_BlendingMode mode) {
      mode_ = mode;
    }
    
    void BlendKernel::set_overlay (const Texture &overlay) {
      overlay_ = overlay;
    }
    
    void BlendKernel::setup (CommandEncoder &encoder) {
      if (!overlay_) return;
      encoder.set(overlay_,2);
      encoder.set(opacity_,3);
      encoder.set(mode_,4);
    }
}