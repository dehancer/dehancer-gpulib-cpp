//
// Created by denn on 18.01.2021.
//

#include "dehancer/gpu/overlays/OverlayKernel.h"

#include <cmath>

#include "dehancer/gpu/Log.h"

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
            overlay_src_(nullptr),
            overlay_base_(nullptr),
            interpolation_mode_(interpolation),
            options_(options),
            overlay_offset_({0.0f,0.0f})
    {
      set_overlay(overlay);
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
      if (overlay_src_!=overlay || overlay_src_ == nullptr) {
        if (overlay) {
          dehancer::log::print(" *** set_overlay  src: %ix%i", overlay->get_width(), overlay->get_height());
        }
        if (overlay_base_) {
          dehancer::log::print(" *** set_overlay base: %ix%i", overlay_base_->get_width(), overlay_base_->get_height());
        }
        overlay_src_ = overlay;
        overlay_base_ = nullptr;
        resize_overlay();
      } else {
        dehancer::log::print(" *** set_overlay OLD src: %ix%i", overlay_src_->get_width(), overlay_src_->get_height());
      }
    }
    
    void OverlayKernel::setup (CommandEncoder &encoder) {
      if (overlay_src_) {
        dehancer::log::print(" *** setup  src: %ix%i", overlay_src_->get_width(), overlay_src_->get_height());
      }
      if (overlay_base_) {
        dehancer::log::print(" *** setup base: %ix%i", overlay_base_->get_width(), overlay_base_->get_height());
      }
      if (!overlay_base_) {
        resize_overlay();
        if (!overlay_base_)
          return;
      }
      encoder.set(overlay_base_,2);
      encoder.set(options_.opacity,3);
      encoder.set(options_.horizontal_flipped,4);
      encoder.set(options_.vertical_flipped,5);
      encoder.set(overlay_offset_,6);
    }
    
    void OverlayKernel::set_interpolation (ResampleKernel::Mode mode) {
      interpolation_mode_ = mode;
    }
    
    void OverlayKernel::set_destination (const Texture &destination) {
      Kernel::set_destination(destination);
      resize_overlay();
    }
    
    void OverlayKernel::resize_overlay () {
      
      auto dest = get_destination();
      
      if (dest){
        
        auto desc = dest->get_desc();
        
        if (overlay_src_) {
          
          auto desc_o = overlay_src_->get_desc();
          
          float scale = std::fmin((float)desc.width/(float)desc_o.width, (float)desc.height/(float)desc_o.height);
          
          desc_o.height = std::floor((float) desc_o.height * scale);
          desc_o.width = std::floor((float) desc_o.width * scale);
          
          auto desc_s = overlay_src_->get_desc();
  
          dehancer::log::print(" *** resize_overlay: scale: %f, base: %ix%i dest: %ix%i", scale, desc_o.width , desc_o.height, desc.width, desc.height);
  
          if (desc_s != desc_o || !overlay_base_) {
            
            overlay_base_ = desc_o.make(get_command_queue());
            
            ResampleKernel(get_command_queue(),
                           overlay_src_, overlay_base_,
                           interpolation_mode_,
                           get_wait_completed()).process();
            
            if (dest->get_width() != overlay_base_->get_width()) {
              overlay_offset_.x() =
                      static_cast<float>(dest->get_width()) - static_cast<float>(overlay_base_->get_width());
            } else {
              overlay_offset_.y() =
                      static_cast<float>(dest->get_height()) - static_cast<float>(overlay_base_->get_height());
            }
            
            overlay_offset_ *= 0.5f;

          }
        }
      }
    }
  
}