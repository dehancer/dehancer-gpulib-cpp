//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "utils.h"

namespace dehancer::cuda {

    class Context {

    public:
        explicit Context(const void *command_queue);
        [[nodiscard]] CUstream get_command_queue() const;
        [[nodiscard]] CUcontext get_command_context() const;
        [[nodiscard]] CUdevice get_device_id() const;
        void get_device_info(cudaDeviceProp& info) const;
        void get_mem_info(size_t& total, size_t& free);
        
        void push() const;
        void pop() const;
        
    private:
        const void *command_queue_;
        mutable CUcontext context_;
        mutable CUdevice device_id_;
    };
}

