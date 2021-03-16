//
// Created by denn on 18.01.2021.
//

#include "dehancer/gpu/overlays/OverlayKernel.h"

namespace dehancer {
    
    OverlayKernel::OverlayKernel (const void *command_queue,
                                  const Texture &base,
                                  const Texture &destination,
                                  const Texture &overlay,
                                  Options options,
                                  ResampleKernel::Mode interpolation,
                                  bool wait_until_completed,
                                  const std::string &library_path):
            Kernel(command_queue, "kernel_overlay_image", base, destination, wait_until_completed, library_path),
            overlay_(overlay),
            interpolation_mode_(interpolation),
            options_(options)
    {
    }
    
    OverlayKernel::OverlayKernel (const void *command_queue,
                                  const Texture &overlay,
                                  Options options,
                                  ResampleKernel::Mode interpolation,
                                  bool wait_until_completed,
                                  const std::string &library_path):
            OverlayKernel(command_queue, nullptr, nullptr,  overlay, options, interpolation, wait_until_completed, library_path)
    {
    }
    
    OverlayKernel::OverlayKernel (const void *command_queue,
                                  Options options,
                                  ResampleKernel::Mode interpolation,
                                  bool wait_until_completed,
                                  const std::string &library_path):
            OverlayKernel(command_queue, nullptr, nullptr, nullptr, options, interpolation, wait_until_completed, library_path)
    {
    }
    
    void OverlayKernel::set_options (Options options) {
      options_ = options;
    }
    
    void OverlayKernel::set_overlay (const Texture &overlay) {
      overlay_ = overlay;
    }
    
    void OverlayKernel::setup (CommandEncoder &encoder) {
      if (!overlay_) return;
      encoder.set(overlay_,2);
      encoder.set(options_.opacity,3);
      encoder.set(interpolation_mode_,4);
      encoder.set(options_.horizontal_flipped,5);
      encoder.set(options_.vertical_flipped,6);
    }
    
    void OverlayKernel::set_interpolation (ResampleKernel::Mode mode) {
      interpolation_mode_ = mode;
    }
  
}