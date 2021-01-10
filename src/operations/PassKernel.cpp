//
// Created by denn nevera on 2019-08-02.
//

#include "dehancer/gpu/operations/PassKernel.h"

namespace dehancer {
    PassKernel::PassKernel(const void *command_queue, const Texture &source, const Texture &destination,
                           bool wait_until_completed, const std::string &library_path) :
            Kernel(command_queue, "kernel_dehancer_pass", source, destination, wait_until_completed, library_path)
    {}
}