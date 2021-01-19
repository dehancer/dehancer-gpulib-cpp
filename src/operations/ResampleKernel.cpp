//
// Created by denn nevera on 2019-08-02.
//

#include "dehancer/gpu/operations/ResampleKernel.h"

namespace dehancer {
    
    static std::vector<std::string> _resample_kernel_name_ = {
            "kernel_bilinear",
            "kernel_bicubic"
    };
    
    ResampleKernel::ResampleKernel(const void *command_queue,
                                   const Texture &source,
                                   const Texture &destination,
                                   Mode mode,
                                   bool wait_until_completed,
                                   const std::string &library_path) :
            Kernel(command_queue,
                   _resample_kernel_name_[mode],
                   source, destination, wait_until_completed, library_path)
    {}
    
    ResampleKernel::ResampleKernel (const void *command_queue,
                                    Mode mode,
                                    bool wait_until_completed,
                                    const std::string &library_path):
            ResampleKernel(command_queue, nullptr, nullptr, mode, wait_until_completed, library_path)
    {}
    
    void ResampleKernel::set_mode (float Mode) {}
}