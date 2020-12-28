//
//  DeviceCache.h
//  XPC Service
//
//  Created by Apple on 1/24/18.
//  Copyright (c) 2019 Apple Inc. All rights reserved.
//

#pragma once

#include "dehancer/gpu/DeviceConfig.h"
#include "dehancer/Common.h"
#include <memory>

namespace dehancer {

    namespace device {

        /***
         * Type of cached acceleration device
         */
        enum Type:int {
            gpu = 1<<1,
            cpu = 1<<2,
            unknown = 1<<3
        };

        typedef int TypeFilter;

        /***
         * Get a device name from handler
         * @param device - platform based device handler
         * @return - device name
         */
        [[nodiscard]] std::string get_name(const void* device);

        /***
         * Get a device id from handler
         * @param device - platform based device handler
         * @return - device identifier
         */
        [[nodiscard]] uint64_t    get_id(const void* device);

        /***
         * Get a device acceleration type
         * @param device - platform based device handler
         * @return - device type
         */
        [[nodiscard]] Type        get_type(const void* device);
    }

    namespace impl {
        struct gpu_device_cache;
    }

    /***
     * GPU device interface API.
     * Can be get from singleton device cache instance: DeviceCache::Instance()
     */
    struct gpu_device_cache {
    public:

        /***
         * Get platform specific list of device handlers
         * @return device list
         */
        virtual std::vector<void *> get_device_list() { return get_device_list(device::Type::gpu | device::Type::cpu);};

        /***
         * Get platform specific list of device handlers
         * @param filter with specific hardware
         * @return
         */
        virtual std::vector<void *> get_device_list(device::TypeFilter filter);

        /***
         * Get a device by id
         * @param device_id - device id
         * @return platform based handler
         */
        virtual void* get_device(uint64_t device_id) ;

        /***
         * Get a default device
         * @return platform based handler
         */
        virtual void* get_default_device() ;

        /***
         * Get command queue binds certain device.
         * It can be command queue, stream or context in term of different platforms.
         * In term dehancer gpu lib we'll call always command queue.
         *
         * @param device_id - device id
         * @return platform based command queue
         */
        virtual void* get_command_queue(uint64_t device_id) ;

        /***
         * Get command queue binds with default device.
         * @return platform based command queue
         */
        virtual void* get_default_command_queue() ;

        /***
         * After use command queue must be returnet to cache
         * @param platform based command queue
         */
        virtual void return_command_queue(const void *queue)  ;

        virtual ~gpu_device_cache() = default;

        friend class Singleton<gpu_device_cache>;
    private:
        gpu_device_cache();
        std::shared_ptr<impl::gpu_device_cache> impl_;
    };

    /***
     * Global Device cache object. This one is created once per process.
     */
    class DeviceCache: public Singleton<gpu_device_cache>{
       public:
           DeviceCache() = default;
       };
}
