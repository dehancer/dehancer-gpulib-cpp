//
// Created by denn nevera on 2019-08-02.
//

#include "dehancer/gpu/operations/Rotate90Kernel.h"

namespace dehancer {
    
    Rotate90Kernel::Rotate90Kernel(const void *command_queue,
                           const Texture &source,
                           const Texture &destination,
                           Mode mode,
                           bool wait_until_completed,
                           const std::string &library_path) :
            Kernel(command_queue,
                   "kernel_rotate90",
                   source, destination, wait_until_completed, library_path),
            mode_(mode) {
    }
    
    Rotate90Kernel::Rotate90Kernel (const void *command_queue,
                            Mode mode,
                            bool wait_until_completed,
                            const std::string &library_path):
            Rotate90Kernel(command_queue, nullptr, nullptr, mode, wait_until_completed, library_path)
    {}
    
    void Rotate90Kernel::set_mode (Mode mode) {
      mode_= mode;
    }
    
    void Rotate90Kernel::setup (CommandEncoder &encoder) {
      encoder.set((int)mode_,2);
    }
}