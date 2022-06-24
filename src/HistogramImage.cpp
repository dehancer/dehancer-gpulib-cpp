//
// Created by denn on 23.06.2022.
//

#include <utility>

#include "dehancer/gpu/kernels/histogram_common.h"
#include "dehancer/gpu/HistogramImage.h"

namespace dehancer {
    
    namespace impl {
        
        struct HistogramImpl {
            HistogramImage*  root;
            math::Histogram histogram;
            Texture         source;
            Memory          partial_histogram_buffer;
            
            explicit HistogramImpl(HistogramImage* root):
            root(root),
            histogram(DEHANCER_HISTOGRAM_CHANNELS,DEHANCER_HISTOGRAM_SIZE),
            source(nullptr),
            partial_histogram_buffer(nullptr)
            {}
    
            [[nodiscard]] const math::Histogram& get_histogram() const {  return histogram; };
            
        };
    
        class HistogramAcc: public dehancer::Function {
        public:
            
            HistogramAcc(const void *command_queue,
                         const Memory& partial_histogram_buffer,
                         size_t size,
                         size_t channels,
                         bool wait_until_completed = true,
                         const std::string &library_path = "");
            
            void process(size_t grid_size, size_t block_size, size_t threads_in_grid);
            
        private:
            size_t size_;
            size_t channels_;
            Memory partial_histogram_buffer_;
            Memory histogram_buffer_;
        };
        
    }
    
    HistogramImage::HistogramImage (
            const void *command_queue,
            const Texture &source,
            bool wait_until_completed,
            const std::string &library_path):
            Function(command_queue, "kernel_histogram_image", wait_until_completed, library_path),
            impl_(std::make_shared<impl::HistogramImpl>(this))
    {
      set_source(source);
    }
    
    const math::Histogram &HistogramImage::get_histogram () const {
      return impl_->get_histogram () ;
    }
    
    void HistogramImage::set_source (const Texture &source) {
      impl_->source = source;
      if (impl_->source) {
        size_t length = (impl_->histogram.get_size().size+1) * impl_->histogram.get_size().num_channels * sizeof (uint);
        auto command_size = ask_compute_size(impl_->source);
        length *= command_size.threads_in_grid;
        impl_->partial_histogram_buffer = MemoryHolder::Make(get_command_queue(),length);
      }
      else {
        impl_->partial_histogram_buffer = nullptr;
      }
    }
    
    const Texture &HistogramImage::get_source () const {
      return impl_->source;
    }
    
    void HistogramImage::process () {
      if (
              !impl_->source
              ||
              !impl_->partial_histogram_buffer
              ) return;
  
      auto workgroup_size = get_block_max_size();
      auto compute_size = Function::ask_compute_size(impl_->source);
      
      execute(compute_size, [this](CommandEncoder& encoder) {
          encoder.set(get_source(),0);
          encoder.set(impl_->partial_histogram_buffer,1);
      });
  
      auto grid_size = impl_->histogram.get_size().size * impl_->histogram.get_size().num_channels;
      auto block_size = (workgroup_size >  impl_->histogram.get_size().size) ?  impl_->histogram.get_size().size : workgroup_size;
  
      impl::HistogramAcc(get_command_queue(),
                         impl_->partial_histogram_buffer,
                         impl_->histogram.get_size().size,
                         impl_->histogram.get_size().num_channels,
                         true, get_library_path())
                         .process(grid_size,block_size,compute_size.threads_in_grid);
    }
    
    namespace impl {
        HistogramAcc::HistogramAcc (const void *command_queue,
                                    const Memory &partial_histogram_buffer,
                                    size_t size,
                                    size_t channels,
                                    bool wait_until_completed,
                                    const std::string &library_path ):
                Function(command_queue, "kernel_sum_partial_histogram_image", wait_until_completed, library_path),
                size_(size),
                channels_(channels),
                partial_histogram_buffer_(partial_histogram_buffer)
        {
        }
    
        void HistogramAcc::process (size_t grid_size, size_t block_size, size_t threads_in_grid) {
          CommandEncoder::ComputeSize size = {
                  .grid =
                  {
                          .width = grid_size,
                          .height = 1,
                          .depth = 1
                  },
                  .block = {
                          .width = block_size,
                          .height = 1,
                          .depth = 1
                  },
                  .threads_in_grid = threads_in_grid
          };
  
          size_t length = (size_+1) * channels_ * sizeof (uint);
          histogram_buffer_ = MemoryHolder::Make(get_command_queue(),length);
  
          execute(size, [this,size](CommandEncoder& encoder) {
              encoder.set(this->partial_histogram_buffer_,0);
              encoder.set((int)size.threads_in_grid,1);
              encoder.set(histogram_buffer_,2);
          });
  
          std::vector<uint> buffer;
          histogram_buffer_->get_contents(buffer);
          for(uint i = 0; i < size_ ; ++i) {
            std::cout << "["<<i<<"] = " << buffer[i] << std::endl;
          }
        }
    }
}