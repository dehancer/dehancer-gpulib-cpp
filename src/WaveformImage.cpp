//
// Created by denn on 23.06.2022.
//

#include <utility>

#include "dehancer/gpu/kernels/waveform_common.h"
#include "dehancer/gpu/WaveformImage.h"
#include "dehancer/gpu/spaces/StreamTransform.h"
#include "dehancer/math.hpp"
using float3=dehancer::math::float3;
using float4=dehancer::math::float4;
#include "dehancer/gpu/kernels/constants.h"

namespace dehancer {
    
    namespace impl {
        
        struct WaveformImpl {
            WaveformImage*  root;
            WaveformImage::Options options;
            math::Waveform waveform;
            Texture         source;
            Memory          partial_waveform_buffer;
            
            explicit WaveformImpl(WaveformImage* root, const WaveformImage::Options& options):
            root(root),
            options(options),
            waveform(DEHANCER_WAVEFORM_CHANNELS,DEHANCER_WAVEFORM_SIZE),
            source(nullptr),
            partial_waveform_buffer(nullptr)
            {}
    
            [[nodiscard]] const math::Waveform& get_waveform() const {  return waveform; };
            
        };
    
        class WaveformAcc: public dehancer::Function {
        public:
            
            WaveformAcc(const void *command_queue,
                         const Memory& partial_waveform_buffer,
                         size_t size,
                         size_t channels,
                         const WaveformImage::Options& options,
                         bool wait_until_completed = true,
                         const std::string &library_path = "");
            
            void process(size_t grid_size, size_t block_size, size_t threads_in_grid);
            
            const  std::vector<std::vector<float>>& get_waveform() const { return waveform_;};
            
        private:
            size_t size_;
            size_t channels_;
            WaveformImage::Options   options_;
            Memory partial_waveform_buffer_;
            Memory waveform_buffer_;
            std::vector<std::vector<float>> waveform_;
        };
        
    }
    
    WaveformImage::WaveformImage (
            const void *command_queue,
            const Texture &source,
            const WaveformImage::Options& options,
            bool wait_until_completed,
            const std::string &library_path):
            Function(command_queue, "kernel_histogram_image", wait_until_completed, library_path),
            impl_(std::make_shared<impl::WaveformImpl>(this, options))
    {
      set_source(source);
    }
    
    const math::Waveform &WaveformImage::get_waveform () const {
      return impl_->get_waveform () ;
    }
    
    void WaveformImage::set_source (const Texture &source) {
      impl_->source = source;
      if (impl_->source) {
        size_t length = (impl_->waveform.get_size().size+1) * impl_->waveform.get_size().num_channels * sizeof (uint);
        auto command_size = ask_compute_size(impl_->source);
        length *= command_size.threads_in_grid;
        MemoryDesc desc = {
                .length = length,
                .mem_flags = static_cast<MemoryDesc::MemFlags>(MemoryDesc::MemFlags::less_memory |
                                                               MemoryDesc::MemFlags::read_write)
        };
        impl_->partial_waveform_buffer = desc.make(get_command_queue());
      }
      else {
        impl_->partial_waveform_buffer = nullptr;
      }
    }
    
    const Texture &WaveformImage::get_source () const {
      return impl_->source;
    }
    
    void WaveformImage::process () {
    
      if (
              !impl_->source
              ||
              !impl_->partial_waveform_buffer
              ) return;
  
      auto workgroup_size = get_block_max_size();
      auto compute_size = Function::ask_compute_size(impl_->source);
  
      auto real_source = get_source();
      
      if (impl_->options.transform.enabled) {
        auto dest = get_source()->get_desc().make(get_command_queue());
        auto transformer = dehancer::StreamTransform(get_command_queue(),
                                                     get_source(),
                                                     dest,
                                                     impl_->options.transform.space,
                                                     impl_->options.transform.direction,
                                                     1.0f,
                                                     true,
                                                     get_library_path()
                                                     );
  
//        transformer.set_impact(1.0f);
//        transformer.set_source(get_source());
//        transformer.set_destination(dest);
        transformer.process();
        real_source = dest;
      }
      
      execute(compute_size, [this,compute_size,&real_source](CommandEncoder& encoder) {
          encoder.set(real_source,0);
          encoder.set(static_cast<int>(impl_->options.luma_type), 1);
          encoder.set(impl_->partial_waveform_buffer,2);
          encoder.set((int)compute_size.threads_in_grid,3);
      });
      
      auto grid_size = impl_->waveform.get_size().size * impl_->waveform.get_size().num_channels;
      auto block_size = (workgroup_size >  impl_->waveform.get_size().size) ?  impl_->waveform.get_size().size : workgroup_size;
  
      auto acc = impl::WaveformAcc(get_command_queue(),
                         impl_->partial_waveform_buffer,
                         impl_->waveform.get_size().size,
                         impl_->waveform.get_size().num_channels,
                         impl_->options,
                         true,
                         get_library_path());

      acc.process(grid_size,block_size,compute_size.threads_in_grid);

      auto& buffer = acc.get_waveform();

      impl_->waveform.update(buffer);
    }
    
    void WaveformImage::set_options (const WaveformImage::Options &options) {
      impl_->options = options;
    }
    
    namespace impl {
        WaveformAcc::WaveformAcc (const void *command_queue,
                                    const Memory &partial_waveform_buffer,
                                    size_t size,
                                    size_t channels,
                                    const WaveformImage::Options& options,
                                    bool wait_until_completed,
                                    const std::string &library_path ):
                Function(command_queue, "kernel_sum_partial_histogram_image", wait_until_completed, library_path),
                size_(size),
                channels_(channels),
                options_(options),
                partial_waveform_buffer_(partial_waveform_buffer)
        {
        }
    
        void WaveformAcc::process (size_t grid_size, size_t block_size, size_t threads_in_grid) {
         
          CommandEncoder::ComputeSize size = {
                  .grid =
                  {
                          #ifdef DEHANCER_GPU_OPENCL
                          .width = grid_size,
                          #else
                          .width = grid_size/block_size,
                          #endif
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
          waveform_buffer_ = MemoryHolder::Make(get_command_queue(),length);
  
          execute(size, [this,size](CommandEncoder& encoder) {
              encoder.set(this->partial_waveform_buffer_,0);
              encoder.set((int)size.threads_in_grid,1);
              encoder.set(waveform_buffer_,2);
          });
  
          std::vector<uint> buffer;
          waveform_buffer_->get_contents(buffer);
  
          waveform_.resize(channels_);
          for (int c = 0; c < (int)channels_; ++c) {
            waveform_[c].resize(size_);
            for (int i = 0; i < (int)size_; ++i) {
              waveform_[c][i] = static_cast<float>(buffer[c*DEHANCER_WAVEFORM_BUFF_SIZE+i]);
            }
            if (options_.edges.ignore) {
              size_t width = floor(options_.edges.left_trim);
              float left_fract = (1.0f - (options_.edges.left_trim - static_cast<float>(width)));
              for (size_t i = 0; i < width && i< size_ ; ++i) {
                waveform_[c][i] = 0;
              }
              if (width < size_){
                waveform_[c][width] *= left_fract;
              }
              width = std::floor(options_.edges.right_trim);
              float right_fract = (1.0f-(options_.edges.right_trim - static_cast<float>(width)));
              int cut = (int)size_-(int)width-1;
              for (int i = (int)size_-1; i >= 0 && i > cut ; --i) {
                waveform_[c][i] = 0;
              }
              if (cut>0)
                waveform_[c][cut] *= right_fract;
            }
          }
        }
    }
}