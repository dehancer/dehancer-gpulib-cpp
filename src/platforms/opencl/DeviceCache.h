//
// Created by denn nevera on 15/11/2020.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/opencl/buffer.h"
#include "dehancer/opencl/embeddedProgram.h"

#include <mutex>
#include <memory>
#include <map>

const size_t  kMaxCommandQueues  = 16;

namespace dehancer::opencl {

    struct gpu_command_queue_item {
        bool in_use = false;
        cl_command_queue command_queue = nullptr;
        explicit gpu_command_queue_item(bool in_use, cl_command_queue command_queue);
        ~gpu_command_queue_item();
    };

    struct gpu_device_item {

        explicit gpu_device_item(const std::shared_ptr<clHelper::Device>&  device);
        ~gpu_device_item();

        cl_command_queue get_next_free_command_queue();
        bool return_command_queue(cl_command_queue command_queue);

        std::shared_ptr<clHelper::Device> device;
        cl_context context = nullptr;
        std::vector<std::shared_ptr<gpu_command_queue_item>> command_queue_cache;
        mutable std::mutex mutex_;
    };

    struct gpu_device_cache {
    public:
        gpu_device_cache();

        virtual void* get_device(const void* id) ;
        virtual void* get_default_device() ;
        virtual void* get_command_queue(const void* id) ;
        virtual void* get_default_command_queue() ;
        virtual void return_command_queue(const void *q)  ;

    private:
        std::vector<std::shared_ptr<gpu_device_item>> device_caches_;
    };
}

