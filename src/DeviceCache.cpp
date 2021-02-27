//
// Created by denn nevera on 15/11/2020.
//

#include "dehancer/gpu/DeviceCache.h"
#include "dehancer/gpu/Command.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "src/platforms/metal/DeviceCache.h"
#elif defined(DEHANCER_GPU_CUDA)
#include "src/platforms/cuda//DeviceCache.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "src/platforms/opencl/DeviceCache.h"
#endif

#ifdef DEHANCER_GPU_PLATFORM

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

    void *gpu_device_cache::get_device(uint64_t device) {
      return impl_->get_device(device);
    }

    void *gpu_device_cache::get_default_device() {
      return impl_->get_default_device();
    }

    void *gpu_device_cache::get_command_queue(uint64_t device) {
      return impl_->get_command_queue(device);
    }

    void *gpu_device_cache::get_default_command_queue() {
      return impl_->get_default_command_queue();
    }

    void gpu_device_cache::return_command_queue(const void *q) {
      impl_->return_command_queue(q);
    }

    std::vector<void *> gpu_device_cache::get_device_list(int filter) {
      return impl_->get_device_list(filter);
    }

    uint64_t device::get_id(const void *device) {
      return DEHANCER_GPU_PLATFORM::device::get_id(device);
    }

    std::string device::get_name(const void *device) {
      return DEHANCER_GPU_PLATFORM::device::get_name(device);
    }

    dehancer::device::Type device::get_type(const void *device) {
      return DEHANCER_GPU_PLATFORM::device::get_type(device);
    }
}

#endif