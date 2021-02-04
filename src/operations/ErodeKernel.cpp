//
// Created by denn on 02.02.2021.
//

#include "dehancer/gpu/operations/ErodeKernel.h"

namespace dehancer {
    ErodeKernel::ErodeKernel (const void *command_queue, const dehancer::Texture &source,
                              const dehancer::Texture &destination, size_t patches, size_t iterations,
                              bool wait_until_completed, const std::string &library_path) :
            MorphKernel(command_queue, "kernel_erode", source, destination, patches, iterations, wait_until_completed,
                        library_path)
    {
    }
    
    ErodeKernel::ErodeKernel (const void *command_queue, size_t patches, size_t iterations, bool wait_until_completed,
                              const std::string &library_path):
            ErodeKernel(command_queue, nullptr, nullptr, patches, iterations, wait_until_completed, library_path)
    {
    }
}
