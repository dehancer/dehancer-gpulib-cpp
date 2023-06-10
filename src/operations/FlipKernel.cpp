//
// Created by denn nevera on 2019-08-02.
//

#include "dehancer/gpu/operations/FlipKernel.h"

namespace dehancer {
    
    FlipKernel::FlipKernel(const void *command_queue,
                           const Texture &source,
                           const Texture &destination,
                           Mode mode,
                           bool wait_until_completed,
                           const std::string &library_path) :
            Kernel(command_queue,
                   "kernel_flip",
                   source, destination, wait_until_completed, library_path),
            mode_(mode) {
    }
    
    FlipKernel::FlipKernel (const void *command_queue,
                            Mode mode,
                            bool wait_until_completed,
                            const std::string &library_path):
            FlipKernel(command_queue, nullptr, nullptr, mode, wait_until_completed, library_path)
    {}
    
    void FlipKernel::set_mode (Mode mode) {
      mode_= mode;
    }
    
    void FlipKernel::setup (CommandEncoder &encoder) {
      encoder.set((bool)((int)mode_&(int)Mode::horizontal),2);
      encoder.set((bool)((int)mode_&(int)Mode::vertical),3);
    }
}