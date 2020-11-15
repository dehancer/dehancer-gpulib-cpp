//
// Created by denn nevera on 15/11/2020.
//

#include "DeviceCache.h"

#include <utility>

namespace dehancer::opencl {

    gpu_device_cache::gpu_device_cache():
            device_caches_()
    {
      auto devices = clHelper::getAllDevices();
      for (auto d: devices){
        device_caches_.push_back(std::make_shared<gpu_device_item>(d));
      }
    }

    void *gpu_device_cache::get_device(const void* id) {
      for (const auto& item: device_caches_) {
        if (item->device && item->device->clDeviceID && item->device->clDeviceID == id) {
          return item->device.get();
        }
      }
      return nullptr;
    }

    void *gpu_device_cache::get_default_device() {
      if (!device_caches_.empty()) {
        auto it = device_caches_[device_caches_.size()-1];
        return it->device.get();
      }
      return nullptr;
    }

    void *gpu_device_cache::get_command_queue(const void* id) {

      if (!id) return nullptr;

      for (const auto& item: device_caches_) {
        if (item->device && item->device->clDeviceID && item->device->clDeviceID == id) {
          auto queue = item->get_next_free_command_queue();
          if (queue) return queue;
        }
      }

      auto devices = clHelper::getAllDevices();
      std::shared_ptr<clHelper::Device> device = nullptr;
      for (const auto& next: devices){
        if(next->clDeviceID && next->clDeviceID == id) {
          device = next; break;
        }
      }

      cl_command_queue q = nullptr;

      if (device) {
        auto item = std::make_shared<gpu_device_item>(device);
        device_caches_.push_back(item);
        q = item->get_next_free_command_queue();
      }

      return q;
    }

    void *gpu_device_cache::get_default_command_queue() {
      void* pointer = get_default_device();
      if (!pointer) return nullptr;
      auto* device = reinterpret_cast<clHelper::Device *>(pointer);
      return get_command_queue(device->clDeviceID);
    }

    void gpu_device_cache::return_command_queue(const void *q) {
      for (auto& device: device_caches_) {
        if(device->return_command_queue(reinterpret_cast<cl_command_queue>((void*)q)))
          break;
      }
    }

    gpu_device_item::gpu_device_item(const std::shared_ptr<clHelper::Device>& device)
            :device(device)
    {
      cl_device_id device_id = device->clDeviceID;
      cl_int ret = 0;
      context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

      if (ret != CL_SUCCESS) {
        throw std::runtime_error("Unable to create new OpenCL context for device: " + device->name);
      }

      for (int i = 0; i < kMaxCommandQueues; ++i) {

#ifdef __APPLE__
        auto q = clCreateCommandQueue(context, device_id, 0, &ret);
#else
        cl_queue_properties devQueueProps[] = {  CL_QUEUE_PROPERTIES,
                                                 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
                                                 |CL_QUEUE_ON_DEVICE
                                                 //|CL_QUEUE_ON_DEVICE_DEFAULT // if we want to use only one queue per device
                ,
                                                 CL_QUEUE_SIZE,
                                                 CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE
                ,
                                                 0 };

        auto q = clCreateCommandQueueWithProperties(context, device_id, devQueueProps, &ret);
#endif

        if (ret != CL_SUCCESS) {
          throw std::runtime_error("Unable to create new OpenCL command queue for device: " + device->name);
        }

        command_queue_cache.push_back(std::make_shared<gpu_command_queue_item>(false,q));
      }
    }

    cl_command_queue gpu_device_item::get_next_free_command_queue() {
      std::lock_guard lock(mutex_);
      for(auto& item: command_queue_cache) {
        if (item->in_use) continue;
        item->in_use = true;
        return item->command_queue;
      }
      return nullptr;
    }

    bool gpu_device_item::return_command_queue(cl_command_queue command_queue) {
      std::lock_guard lock(mutex_);
      bool found = false;
      for(auto& item: command_queue_cache) {
        if (item->command_queue!=command_queue) continue;
        found = true;
        if (item->in_use) {
          item->in_use = false;
        }
      }
      return found;
    }

    gpu_device_item::~gpu_device_item() {
      std::cerr << " ~gpu_device_item("<<context<<")" << std::endl;
      if (context)
        clReleaseContext(context);
    }

    gpu_command_queue_item::~gpu_command_queue_item() {
      std::cerr << " ~gpu_command_queue_item("<<command_queue<<")" << std::endl;
      if (command_queue)
        clReleaseCommandQueue(command_queue);
    }

    gpu_command_queue_item::gpu_command_queue_item(bool in_use_, cl_command_queue command_queue_) {
      in_use = in_use_;
      command_queue = command_queue_;
    }
}