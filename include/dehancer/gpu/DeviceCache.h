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

    namespace impl {
        struct gpu_device_cache;
    }

    struct gpu_device_cache {
    public:
        gpu_device_cache();

        virtual void* get_device(const void* id) ;
        virtual void* get_default_device() ;
        virtual void* get_command_queue(const void* id) ;
        virtual void* get_default_command_queue() ;
        virtual void return_command_queue(const void *q)  ;

        virtual ~gpu_device_cache() = default;
        
    private:
        std::shared_ptr<impl::gpu_device_cache> impl_;
    };

    class DeviceCache: public Singleton<gpu_device_cache>{
       public:
           DeviceCache() = default;
       };
}
