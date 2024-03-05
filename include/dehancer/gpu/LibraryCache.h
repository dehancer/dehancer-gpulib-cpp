//
// Created by dimich on 05.03.2024.
//

#ifndef DEHANCER_GPULIB_LIBRARYCACHE_H
#define DEHANCER_GPULIB_LIBRARYCACHE_H

#include "dehancer/gpu/DeviceConfig.h"
#include "dehancer/Common.h"
#include <memory>

#if defined(DEHANCER_GPU_OPENCL)

namespace dehancer {

    namespace impl {
        struct gpu_library_cache;
    }

    struct gpu_library_cache {
    public:
        virtual bool has_cache_for_device(const void *command, uint64_t device_id,
                                  const std::string &library_source = "");

        virtual bool compile_program_for_device(const void *command, uint64_t device_id,
                                        const std::string &library_source = "");

#if defined(DEHANCER_CONTROLLED_SINGLETON)
        friend class ControlledSingleton<gpu_device_cache>;
#else
        friend class SimpleSingleton<gpu_library_cache>;
#endif
    private:
        gpu_library_cache();
        std::shared_ptr<impl::gpu_library_cache> impl_;
    };

    class LibraryCache : public SimpleSingleton<gpu_library_cache> {
    public:
    public:
        LibraryCache() = default;
    };
}
#endif

#endif //DEHANCER_GPULIB_LIBRARYCACHE_H
