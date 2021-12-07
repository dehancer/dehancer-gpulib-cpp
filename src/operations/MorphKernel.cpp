//
// Created by denn on 02.02.2021.
//

#include "dehancer/gpu/operations/MorphKernel.h"

namespace dehancer {
    
    MorphKernel::MorphKernel (const void *command_queue,
                              const std::string& morph_kernel_name,
                              const Texture &source,
                              const Texture &destination,
                              size_t patches,
                              size_t iterations,
                              bool wait_until_completed,
                              const std::string &library_path):
            Kernel(command_queue, morph_kernel_name, source, destination, wait_until_completed, library_path),
            patches_(patches),
            iterations_(iterations)
    {
    }
    
    void MorphKernel::set_size (size_t patches) {
      patches_ = patches;
    }
    
    void MorphKernel::set_iterations (size_t iterations) {
      iterations_ = iterations;
    }
    
    void MorphKernel::process () {
      auto src = get_source();
      
      auto desc = get_destination()->get_desc();
      desc.pixel_format = TextureDesc::PixelFormat::rgba16float;
      //desc.mem_flags = TextureDesc::MemFlags::less_memory;
      
      auto tmp_ = desc.make(get_command_queue());
      
      for(size_t i=0; i<iterations_; ++i){
        execute([this, &src, &tmp_] (CommandEncoder &encoder) {
            encoder.set(src, 0);
            encoder.set(tmp_, 1);
            encoder.set((int) patches_, 2);
            dehancer::math::int2 step = {1, 0};
            encoder.set(step, 3);
            return CommandEncoder::Size::From(tmp_);
        });
        
        execute([this, &tmp_] (CommandEncoder &encoder) {
            encoder.set(tmp_, 0);
            encoder.set(get_destination(), 1);
            encoder.set((int) patches_, 2);
            dehancer::math::int2 step = {0, 1};
            encoder.set(step, 3);
            return CommandEncoder::Size::From(get_destination());
        });
        src = get_destination();
      }
    }
    
    void MorphKernel::set_destination (const Texture &destination) {
      Kernel::set_destination(destination);
    }
    
    void MorphKernel::process (const Texture &source, const Texture &destination) {
      Kernel::process(source, destination);
    }
}