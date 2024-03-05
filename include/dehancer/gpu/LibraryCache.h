//
// Created by dimich on 05.03.2024.
//

#ifndef DEHANCER_GPULIB_LIBRARYCACHE_H
#define DEHANCER_GPULIB_LIBRARYCACHE_H

#include "dehancer/gpu/DeviceConfig.h"
#include "dehancer/gpu/Command.h"
#include "dehancer/Common.h"
#include <memory>

#if defined(DEHANCER_GPU_OPENCL)

namespace dehancer {

    namespace impl {
        class GPULibraryCache;
    }

    class GPULibraryCache : public Command {
    public:
        explicit GPULibraryCache(const void *command_queue);

        virtual bool has_cache(const std::string &library_source = "");

        virtual bool compile_program(const std::string &library_source = "");

    private:
        GPULibraryCache() = default;

        std::shared_ptr<impl::GPULibraryCache> impl_;
    };
}
#endif

#endif //DEHANCER_GPULIB_LIBRARYCACHE_H
