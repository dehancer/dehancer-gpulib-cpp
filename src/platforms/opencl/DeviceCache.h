//
// Created by denn nevera on 15/11/2020.
//

#pragma once

#include "dehancer/opencl/device.h"
#include "dehancer/gpu/DeviceCache.h"

#include <cstddef>
#include <mutex>
#include <memory>

const size_t kMaxCommandQueues  = 16;

namespace dehancer::opencl {

    namespace device {
        [[nodiscard]] std::string get_name(const void* device);
        [[nodiscard]] uint64_t    get_id(const void* device);
        [[nodiscard]] dehancer::device::Type get_type(const void* device);
    }

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

       std::vector<void *> get_device_list(dehancer::device::TypeFilter filter);
       void* get_device(uint64_t id) ;
       void* get_default_device() ;
       void* get_command_queue(uint64_t id) ;
       void* get_default_command_queue() ;
       void return_command_queue(const void *q)  ;

    private:
        std::vector<std::shared_ptr<gpu_device_item>> device_caches_;
        std::vector<std::shared_ptr<clHelper::Device>> devices_;
    };
}

