//
// Created by denn on 23.06.2022.
//

#include "dehancer/gpu/HistogramKernel.h"

namespace dehancer {
    
    namespace impl {
        struct HistogramKernelImpl {
            HistogramKernel* root;
            math::Histogram histogram;
            Memory          partial_histogram_buffer;
            
            explicit HistogramKernelImpl(HistogramKernel* root): root(root)
            {
              partial_histogram_buffer = MemoryHolder::Make(root->get_command_queue(),1);
            }
    
            [[nodiscard]] const math::Histogram& get_histogram() const {  return histogram; };
  
            
        };
    }
    
    HistogramKernel::HistogramKernel (
            const void *command_queue,
            const Texture &source,
            bool wait_until_completed,
            const std::string &library_path):
            Kernel(command_queue, "kernel_histogram_image", source, nullptr, wait_until_completed, library_path),
            impl_(std::make_shared<impl::HistogramKernelImpl>(this))
    {
    }
    
    const math::Histogram &HistogramKernel::get_histogram () const {
      return impl_->get_histogram () ;
    }
    
    void HistogramKernel::setup (CommandEncoder &encoder) {
      encoder.set(get_source(),0);
      encoder.set(impl_->partial_histogram_buffer,1);
    }
}