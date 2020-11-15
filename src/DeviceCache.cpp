//
// Created by denn nevera on 15/11/2020.
//

#include "dehancer/gpu/DeviceCache.h"

#include "dehancer/gpu/Command.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "src/platforms/metal/DeviceCache.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "src/platforms/opencl/DeviceCache.h"
#endif

namespace dehancer {

    namespace impl {
        class gpu_device_cache: public dehancer::DEHANCER_GPU_PLATFORM::gpu_device_cache {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::gpu_device_cache::gpu_device_cache;
        };
    }

    gpu_device_cache::gpu_device_cache():
            impl_(std::make_shared<impl::gpu_device_cache>())
    {}

    void *gpu_device_cache::get_device(const void* id) {
      return impl_->get_device(id);
    }

    void *gpu_device_cache::get_default_device() {
      return impl_->get_default_device();
    }

    void *gpu_device_cache::get_command_queue(const void* id) {
      return impl_->get_command_queue(id);
    }

    void *gpu_device_cache::get_default_command_queue() {
      return impl_->get_default_command_queue();
    }

    void gpu_device_cache::return_command_queue(const void *q) {
      impl_->return_command_queue(q);
    }
}