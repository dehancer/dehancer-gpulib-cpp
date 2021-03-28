//
// Created by denn on 18.01.2021.
//

#include "dehancer/gpu/overlays/OverlayKernel.h"

#include <cmath>

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
            overlay_(nullptr),
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
      overlay_ = overlay;
      resize_overlay();
    }
    
    void OverlayKernel::setup (CommandEncoder &encoder) {
      if (!overlay_) return;
      encoder.set(overlay_,2);
      encoder.set(options_.opacity,3);
      encoder.set(interpolation_mode_,4);
      encoder.set(options_.horizontal_flipped,5);
      encoder.set(options_.vertical_flipped,6);
      encoder.set(overlay_offset_,7);
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
        if (overlay_) {
          auto desc_o = overlay_->get_desc();
          float scale = std::fmin((float)desc.width/(float)desc_o.width, (float)desc.height/(float)desc_o.height);
          if (scale<1 || scale > 1) {
            
            desc_o.height = std::floor((float )desc_o.height*scale);
            desc_o.width = std::floor((float )desc_o.width*scale);
            
            auto overlay_tmp = desc_o.make(get_command_queue());
            ResampleKernel(get_command_queue(),
                           overlay_, overlay_tmp,
                           interpolation_mode_,
                           get_wait_completed()).process();
            overlay_ = overlay_tmp;
            
            //overlay_offset_.x() =
            if (abs(desc_o.width-desc.width)>0) {
              //overlay_offset_.x() = (float)(desc_o.width-desc.width)/2;
            }
            else {
              //overlay_offset_.y() = (float)(desc_o.height-desc.height)/2;
            }
            
            std::cout << " RESIZE OVERLAY = " << scale << std::endl
                      << "       offset: " << overlay_offset_.x() << " : " << overlay_offset_.y() << std::endl
                      << " overlay size: " << overlay_->get_width() << " : " << overlay_->get_height() << std::endl
                      << "         size: " << dest->get_width() << " : " << dest->get_height() << std::endl
                      << std::endl;
            
          }
        }
      }
    }
  
}