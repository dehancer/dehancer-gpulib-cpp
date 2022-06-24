//
// Created by denn on 23.06.2022.
//

#include <utility>

#include "dehancer/gpu/HistogramImage.h"

namespace dehancer {
    
    namespace impl {
        struct HistogramImpl {
            HistogramImage*  root;
            Texture         source;
            math::Histogram histogram;
            Memory          partial_histogram_buffer;
            
            explicit HistogramImpl(HistogramImage* root, Texture source):
            root(root), source(std::move(source))
            {
              partial_histogram_buffer = MemoryHolder::Make(root->get_command_queue(),1);
            }
    
            [[nodiscard]] const math::Histogram& get_histogram() const {  return histogram; };
  
            
        };
    }
    
    HistogramImage::HistogramImage (
            const void *command_queue,
            const Texture &source,
            bool wait_until_completed,
            const std::string &library_path):
            //Kernel(command_queue, "kernel_histogram_image", source, nullptr, wait_until_completed, library_path),
            Function(command_queue, "kernel_histogram_image",wait_until_completed, library_path),
            impl_(std::make_shared<impl::HistogramImpl>(this, source))
    {
    }
    
    const math::Histogram &HistogramImage::get_histogram () const {
      return impl_->get_histogram () ;
    }
    
    void HistogramImage::set_source (const Texture &source) {
      impl_->source = source;
    }
    
    const Texture &HistogramImage::get_source () const {
      return impl_->source;
    }
    
    void HistogramImage::process () {
      if (!impl_->source) return;
      auto compute_size = Function::ask_compute_size(impl_->source);
      execute(compute_size, [this](CommandEncoder& encoder) {
          encoder.set(get_source(),0);
          encoder.set(impl_->partial_histogram_buffer,1);
      });
    }


//    void HistogramKernel::setup (CommandEncoder &encoder) {
//      encoder.set(get_source(),0);
//      encoder.set(impl_->partial_histogram_buffer,1);
//    }
}