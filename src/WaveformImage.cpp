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
            Memory          waveform_buffer;
            
            explicit WaveformImpl(WaveformImage* root, const WaveformImage::Options& options):
            root(root),
            options(options),
            waveform(DEHANCER_WAVEFORM_CHANNELS,DEHANCER_WAVEFORM_SIZE),
            source(nullptr),
            waveform_buffer(nullptr)
            {}
    
            [[nodiscard]] const math::Waveform& get_waveform() const {  return waveform; };
            
        };
    }

    WaveformImage::WaveformImage (
            const void *command_queue,
            const Texture &source,
            const WaveformImage::Options& options,
            bool wait_until_completed,
            const std::string &library_path):
            Function(command_queue, "kernel_waveform_image", wait_until_completed, library_path),
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
        size_t length = impl_->waveform.get_size().size * impl_->waveform.get_size().num_channels * sizeof (float);
        MemoryDesc desc = {
                .length = length,
                .mem_flags = static_cast<MemoryDesc::MemFlags>(MemoryDesc::MemFlags::less_memory |
                                                               MemoryDesc::MemFlags::read_write)
        };
        impl_->waveform_buffer = desc.make(get_command_queue());
      }
      else {
        impl_->waveform_buffer = nullptr;
      }
    }
    
    const Texture &WaveformImage::get_source () const {
      return impl_->source;
    }
    
    void WaveformImage::process () {
    
      if (
              !impl_->source
              ||
              !impl_->waveform_buffer
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
  
        transformer.process();
        real_source = dest;
      }
      
      execute(compute_size, [this,compute_size,&real_source](CommandEncoder& encoder) {
          encoder.set(real_source,0);
          encoder.set(static_cast<int>(impl_->options.luma_type), 1);
          encoder.set(impl_->waveform_buffer,2);
      });

      std::vector<float4> buffer;
      impl_->waveform_buffer->get_contents(buffer);

      impl_->waveform.update(buffer);
    }
    
    void WaveformImage::set_options (const WaveformImage::Options &options) {
      impl_->options = options;
    }
}