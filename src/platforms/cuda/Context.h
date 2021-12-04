//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "utils.h"
#include <mutex>
#include <map>

namespace dehancer::cuda {

    class Context {

    public:
        
        struct device_ref {
            CUcontext context = nullptr;
            CUdevice device_id = 0;
            bool is_half_texture_allowed = false;
            size_t max_device_threads = 16;
        };
        
    public:
        explicit Context(const void *command_queue);
        [[nodiscard]] CUstream get_command_queue() const;
        [[nodiscard]] CUcontext get_command_context() const;
        [[nodiscard]] CUdevice get_device_id() const;
        void get_device_info(cudaDeviceProp& info) const;
        [[nodiscard]] size_t get_max_threads() const;
        void get_mem_info(size_t& total, size_t& free);
        [[nodiscard]] bool is_half_texture_allowed() const;
    
        void push() const;
        void pop() const;
        
    private:
        const void *command_queue_;
        mutable device_ref device_ref_;
        static std::mutex mutex_;
        static std::map<size_t,device_ref> cache_;
    };
}

