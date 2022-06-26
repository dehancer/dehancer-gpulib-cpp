//
// Created by denn on 23.06.2022.
//

#include <utility>

#include "dehancer/gpu/kernels/histogram_common.h"
#include "dehancer/gpu/HistogramImage.h"
#include "dehancer/math.hpp"
using float3=dehancer::math::float3;
using float4=dehancer::math::float4;
#include "dehancer/gpu/kernels/constants.h"

namespace dehancer {
    
    namespace impl {
        
        struct HistogramImpl {
            HistogramImage*  root;
            HistogramImage::Options options;
            math::Histogram histogram;
            Texture         source;
            Memory          partial_histogram_buffer;
            
            explicit HistogramImpl(HistogramImage* root, const HistogramImage::Options& options):
            root(root),
            options(options),
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
                         const HistogramImage::Options& options,
                         bool wait_until_completed = true,
                         const std::string &library_path = "");
            
            void process(size_t grid_size, size_t block_size, size_t threads_in_grid);
            
            const  std::vector<std::vector<float>>& get_histogram() const { return histogram_;};
            
        private:
            size_t size_;
            size_t channels_;
            HistogramImage::Options   options_;
            Memory partial_histogram_buffer_;
            Memory histogram_buffer_;
            std::vector<std::vector<float>> histogram_;
        };
        
    }
    
    HistogramImage::HistogramImage (
            const void *command_queue,
            const Texture &source,
            const HistogramImage::Options& options,
            bool wait_until_completed,
            const std::string &library_path):
            Function(command_queue, "kernel_histogram_image", wait_until_completed, library_path),
            impl_(std::make_shared<impl::HistogramImpl>(this, options))
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
        MemoryDesc desc = {
                .length = length,
                .mem_flags = static_cast<MemoryDesc::MemFlags>(MemoryDesc::MemFlags::less_memory |
                                                               MemoryDesc::MemFlags::read_write)
        };
        impl_->partial_histogram_buffer = desc.make(get_command_queue());
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
      
      execute(compute_size, [this,compute_size](CommandEncoder& encoder) {
          encoder.set(get_source(),0);
          encoder.set(impl_->partial_histogram_buffer,1);
          encoder.set((int)compute_size.threads_in_grid,2);
      });
      
      auto grid_size = impl_->histogram.get_size().size * impl_->histogram.get_size().num_channels;
      auto block_size = (workgroup_size >  impl_->histogram.get_size().size) ?  impl_->histogram.get_size().size : workgroup_size;
  
      auto acc = impl::HistogramAcc(get_command_queue(),
                         impl_->partial_histogram_buffer,
                         impl_->histogram.get_size().size,
                         impl_->histogram.get_size().num_channels,
                         impl_->options,
                         true,
                         get_library_path());
      
      acc.process(grid_size,block_size,compute_size.threads_in_grid);
      
      auto& buffer = acc.get_histogram();
      
      impl_->histogram.update(buffer);
    }
    
    void HistogramImage::set_options (const HistogramImage::Options &options) {
      impl_->options = options;
    }
    
    namespace impl {
        HistogramAcc::HistogramAcc (const void *command_queue,
                                    const Memory &partial_histogram_buffer,
                                    size_t size,
                                    size_t channels,
                                    const HistogramImage::Options& options,
                                    bool wait_until_completed,
                                    const std::string &library_path ):
                Function(command_queue, "kernel_sum_partial_histogram_image", wait_until_completed, library_path),
                size_(size),
                channels_(channels),
                options_(options),
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
  
          histogram_.resize(channels_);
          for (int c = 0; c < channels_; ++c) {
            histogram_[c].resize(size_);
            for (int i = 0; i < size_; ++i) {
              histogram_[c][i] = static_cast<float>(buffer[c*DEHANCER_HISTOGRAM_BUFF_SIZE+i]);
            }
            if (options_.ignore_edges) {
              histogram_[c][0] = 0;
              histogram_[c][size_-1] = 0;
            }
          }
        }
    }
}