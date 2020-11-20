//
//  DeviceCache.h
//  XPC Service
//
//  Created by Apple on 1/24/18.
//  Copyright (c) 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "dehancer/gpu/DeviceConfig.h"
#include "dehancer/gpu/DeviceCache.h"
#include "dehancer/Common.h"

namespace dehancer::metal {

    namespace device {
        [[nodiscard]] std::string get_name(const void* device);
        [[nodiscard]] uint64_t    get_id(const void* device);
        [[nodiscard]] dehancer::device::Type  get_type(const void* device);
    }

    struct gpu_device_cache {
    public:
        gpu_device_cache();

        std::vector<void *> get_device_list();
        virtual void* get_device(uint64_t id) ;
        virtual void* get_default_device() ;
        virtual void* get_command_queue(uint64_t id) ;
        virtual void* get_default_command_queue() ;
        virtual void return_command_queue(const void *q)  ;

        virtual ~gpu_device_cache() = default;
        
    private:
        void* device_cache_;
    };

    class DeviceCache: public Singleton<gpu_device_cache>{
       public:
           DeviceCache() = default;
       };
}
