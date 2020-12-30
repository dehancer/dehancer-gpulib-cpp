//
// Created by denn nevera on 15/11/2020.
//

#pragma once

#define DEHANCER_GPU_PLATFORM cuda

#include "dehancer/Common.h"
#include "dehancer/gpu/DeviceCache.h"
#include "dehancer/gpu/kernels/cuda/utils.h"

#include <mutex>
#include <memory>
#include <map>

const size_t  kMaxCommandQueues  = 16;

namespace dehancer::cuda {

    struct device_helper {
        CUdevice device_id=-1;
        cudaDeviceProp props{};

        explicit device_helper(CUdevice id);
    };

    namespace device {
        [[nodiscard]] std::string get_name(const void* device);
        [[nodiscard]] uint64_t    get_id(const void* device);
        [[nodiscard]] dehancer::device::Type get_type(const void* device);
    }

    struct gpu_command_queue_item {
        bool in_use = false;
        CUstream command_queue = nullptr;
        explicit gpu_command_queue_item(bool in_use, CUstream command_queue);
    };

    struct gpu_device_item {

        explicit gpu_device_item(const CUdevice&  device);
        ~gpu_device_item();

        CUstream get_next_free_command_queue();
        bool return_command_queue(CUstream command_queue);

        std::shared_ptr<device_helper> device = nullptr;
        CUcontext context = nullptr;
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

        ~gpu_device_cache() = default;

    private:
        std::vector<std::shared_ptr<gpu_device_item>> device_caches_;
        CUdevice default_device_index_;
    };
}

