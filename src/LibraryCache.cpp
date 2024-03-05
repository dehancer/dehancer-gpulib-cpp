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

        class Command : public dehancer::DEHANCER_GPU_PLATFORM::Command {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Command::Command;
        };
    }

    gpu_library_cache::gpu_library_cache() :
            impl_(std::make_shared<impl::gpu_library_cache>()) {}

    bool gpu_library_cache::has_cache(const void *command,
                                      const std::string &library_source) {

        opencl::Command cmd(command, true);
        impl_->has_cache(&cmd, library_source);

    }

    bool gpu_library_cache::compile_program(const void *command,
                                            const std::string &library_source) {
        opencl::Command cmd(command, true);
        impl_->compile_program(&cmd, library_source);

    }

}

#endif
#endif
