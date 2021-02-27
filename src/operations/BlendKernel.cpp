//
// Created by denn on 18.01.2021.
//

#include "dehancer/gpu/operations/BlendKernel.h"

namespace dehancer {
    
    BlendKernel::BlendKernel (const void *command_queue,
                              const Texture &base,
                              const Texture &destination,
                              const Texture &overlay,
                              const Texture &mask,
                              float opacity,
                              Mode mode,
                              ResampleKernel::Mode interpolation,
                              bool wait_until_completed,
                              const std::string &library_path):
            Kernel(command_queue, "kernel_blend", base, destination, wait_until_completed, library_path),
            overlay_(overlay),
            mask_(mask),
            opacity_(opacity),
            mode_(mode),
            interpolation_mode_(interpolation)
    {
      has_mask_ = !(mask_ == nullptr);
      if (!has_mask_) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
    }
    
    BlendKernel::BlendKernel (const void *command_queue,
                              const Texture &overlay,
                              const Texture &mask,
                              float opacity,
                              Mode mode,
                              ResampleKernel::Mode interpolation,
                              bool wait_until_completed,
                              const std::string &library_path):
            BlendKernel(command_queue, nullptr, nullptr,  overlay, mask, opacity, mode, interpolation, wait_until_completed, library_path)
    {
    }
    
    BlendKernel::BlendKernel (const void *command_queue,
                              const Texture &overlay,
                              float opacity,
                              Mode mode,
                              ResampleKernel::Mode interpolation,
                              bool wait_until_completed,
                              const std::string &library_path):
            BlendKernel(command_queue, nullptr, nullptr, overlay, nullptr, opacity, mode, interpolation, wait_until_completed, library_path)
    {
    }
    
    BlendKernel::BlendKernel (const void *command_queue,
                              float opacity,
                              BlendKernel::Mode mode,
                              ResampleKernel::Mode interpolation,
                              bool wait_until_completed,
                              const std::string &library_path):
            BlendKernel(command_queue, nullptr, nullptr, nullptr, nullptr, opacity, mode, interpolation, wait_until_completed, library_path)
    {
    }
    
    void BlendKernel::set_opacity (float opacity) {
      opacity_ = opacity;
    }
    
    void BlendKernel::set_mode (Mode mode) {
      mode_ = mode;
    }
    
    void BlendKernel::set_overlay (const Texture &overlay) {
      overlay_ = overlay;
    }
    
    void BlendKernel::setup (CommandEncoder &encoder) {
      if (!overlay_) return;
      encoder.set(overlay_,2);
      bool has_mask = !(mask_ == nullptr);
      encoder.set(has_mask,3);
      encoder.set(opacity_,4);
      encoder.set(mode_,5);
      encoder.set(interpolation_mode_,6);
      if (mask_)
        encoder.set(mask_,7);
    }
    
    void BlendKernel::set_interpolation (ResampleKernel::Mode mode) {
      interpolation_mode_ = mode;
    }
    
    void BlendKernel::set_mask (const Texture &mask) {
      mask_ = mask;
      has_mask_ = !(mask_ == nullptr);
      if (!has_mask_) {
        TextureDesc desc ={
          .width = 1,
          .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
    }
  
}