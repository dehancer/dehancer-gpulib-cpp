//
// Created by denn on 02.02.2021.
//

#include "dehancer/gpu/operations/DilateKernel.h"

namespace dehancer {
    DilateKernel::DilateKernel (const void *command_queue, const dehancer::Texture &source,
                                const dehancer::Texture &destination, size_t patches, size_t iterations,
                                bool wait_until_completed, const std::string &library_path) :
            MorphKernel(command_queue, "kernel_dilate", source, destination, patches, iterations, wait_until_completed,
                        library_path) {
    }
    
    DilateKernel::DilateKernel (const void *command_queue, size_t patches, size_t iterations, bool wait_until_completed,
                              const std::string &library_path):
            DilateKernel(command_queue, nullptr, nullptr, patches, iterations, wait_until_completed, library_path)
    {
    }
}
