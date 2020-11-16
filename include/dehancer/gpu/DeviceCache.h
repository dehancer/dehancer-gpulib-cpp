//
//  DeviceCache.h
//  XPC Service
//
//  Created by Apple on 1/24/18.
//  Copyright (c) 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "GpuConfig.h"
#include "dehancer/Common.h"
#include <memory>

namespace dehancer {

    namespace device {

        enum class Type:int {
            gpu = 0,
            cpu,
            unknown
        };

        [[nodiscard]] std::string get_name(const void* device);
        [[nodiscard]] uint64_t    get_id(const void* device);
        [[nodiscard]] Type        get_type(const void* device);
    }

    namespace impl {
        struct gpu_device_cache;
    }

    struct gpu_device_cache {
    public:
        gpu_device_cache();

        virtual std::vector<void *> get_device_list();
        virtual void* get_device(uint64_t device_id) ;
        virtual void* get_default_device() ;
        virtual void* get_command_queue(uint64_t device_id) ;
        virtual void* get_default_command_queue() ;
        virtual void return_command_queue(const void *queue)  ;

        virtual ~gpu_device_cache() = default;
        
    private:
        std::shared_ptr<impl::gpu_device_cache> impl_;
    };

    class DeviceCache: public Singleton<gpu_device_cache>{
       public:
           DeviceCache() = default;
       };
}
