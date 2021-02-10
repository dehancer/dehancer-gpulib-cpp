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
            iterations_(iterations),
            tmp_(destination ? destination->get_desc().make(command_queue) : nullptr)
    {
    }
    
    void MorphKernel::set_size (size_t patches) {
      patches_ = patches;
    }
    
    void MorphKernel::set_iterations (size_t iterations) {
      iterations_ = iterations;
    }
    
    void MorphKernel::process () {
      //process(get_source(), get_destination());
      auto src = get_source();
      for(size_t i=0; i<iterations_; ++i){
        execute([this, &src] (CommandEncoder &encoder) {
            encoder.set(src, 0);
            encoder.set(tmp_, 1);
            encoder.set((int) patches_, 2);
            dehancer::math::int2 step = {1, 0};
            encoder.set(step, 3);
            return CommandEncoder::Size::From(tmp_);
        });
        
        execute([this] (CommandEncoder &encoder) {
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
      if (destination) tmp_ = destination->get_desc().make(get_command_queue());
      else tmp_ = nullptr;
    }
}