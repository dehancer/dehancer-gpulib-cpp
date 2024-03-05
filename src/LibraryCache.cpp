//
// Created by dimich on 05.03.2024.
//

#include <dehancer/gpu/LibraryCache.h>
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_OPENCL)
#ifdef DEHANCER_GPU_PLATFORM

#include "platforms/opencl/LibraryCache.h"
#include "platforms/opencl/Command.h"

namespace dehancer {
    namespace impl {
        class gpu_library_cache : public dehancer::DEHANCER_GPU_PLATFORM::gpu_library_cache {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::gpu_library_cache::gpu_library_cache;
        };

        class Command: public dehancer::DEHANCER_GPU_PLATFORM::Command {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Command::Command;
        };
    }

    gpu_library_cache::gpu_library_cache() :
            impl_(std::make_shared<impl::gpu_library_cache>()) {}

    bool gpu_library_cache::has_cache_for_device(const void *command, uint64_t device_id,
                                                 const std::string &library_source) {

        opencl::Command cmd(command, true);
        impl_->has_cache_for_device(&cmd, device_id, library_source);

    }

    bool gpu_library_cache::compile_program_for_device(const void *command, uint64_t device_id,
                                                       const std::string &library_source) {
        opencl::Command cmd(command, true);
        impl_->compile_program_for_device(&cmd, device_id, library_source);

    }

}

#endif
#endif
