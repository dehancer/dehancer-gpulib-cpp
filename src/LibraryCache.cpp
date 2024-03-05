//
// Created by dimich on 05.03.2024.
//

#include <dehancer/gpu/LibraryCache.h>
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_OPENCL)
#ifdef DEHANCER_GPU_PLATFORM

#include "platforms/opencl/LibraryCache.h"

namespace dehancer {
    namespace impl {
        class GPULibraryCache : public dehancer::DEHANCER_GPU_PLATFORM::GPULibraryCache {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::GPULibraryCache::GPULibraryCache;
        };

        class Command : public dehancer::DEHANCER_GPU_PLATFORM::Command {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Command::Command;
        };
    }

    GPULibraryCache::GPULibraryCache(const void *command_queue)
    :  Command(command_queue, true)
    , impl_(std::make_shared<impl::GPULibraryCache>(Command::impl_.get())) {

    }

    bool GPULibraryCache::has_cache(const std::string &library_source) {
        return impl_->has_cache(library_source);
    }

    bool GPULibraryCache::compile_program( const std::string &library_source) {
        return impl_->compile_program(library_source);
    }

}

#endif
#endif
