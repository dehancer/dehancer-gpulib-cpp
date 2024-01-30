//
// Created by denn nevera on 15/11/2020.
//

#include "DeviceCache.h"
#include <utility>

namespace dehancer::opencl {

    namespace device {
        std::string get_name(const void* id) {
          auto* device = reinterpret_cast<clHelper::Device *>((void *)id);
          if (!device)
            return "unknown";
          std::string name = device->vendor;
          name.append(": "); name.append(device->name);
          return name;
        }

        uint64_t    get_id(const void* id) {
          auto* device = reinterpret_cast<clHelper::Device *>((void *)id);
          if (!device)
            return UINT64_MAX;
          return (uint64_t)(device->clDeviceID);
        }

        dehancer::device::Type get_type(const void* id){
          auto* device = reinterpret_cast<clHelper::Device *>((void *)id);
          if (!device)
            return dehancer::device::Type::unknown;

          cl_device_type device_type;

          clGetDeviceInfo(device->clDeviceID,
                          CL_DEVICE_TYPE,
                          sizeof(device_type), &device_type,
                          nullptr);

          switch (device_type) {
            case CL_DEVICE_TYPE_CPU:
              return dehancer::device::Type::cpu;
            case CL_DEVICE_TYPE_GPU:
              return dehancer::device::Type::gpu;
            default:
              return dehancer::device::Type::unknown;
          }
        }
    }

    gpu_device_cache::gpu_device_cache():
            device_caches_(),
            devices_(clHelper::getAllDevices())
    {
      for (auto d: devices_){
        device_caches_.push_back(std::make_shared<gpu_device_item>(d));
      }
    }

    std::vector<void *> gpu_device_cache::get_device_list(dehancer::device::TypeFilter filter) {
      std::vector<void *> list;
      for(const auto& d: devices_){
        if (filter & device::get_type(d.get()))
         list.push_back(d.get());
      }
      return list;
    }

    void *gpu_device_cache::get_device(uint64_t id) {

      for (const auto& item: device_caches_) {
        auto device = item->device;
        //if (device && device->clDeviceID && device::get_id(device->clDeviceID) == id) {
        if (device && device->clDeviceID && device::get_id((const void *)device.get()) == id) {
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

    void *gpu_device_cache::get_command_queue(uint64_t id) {

      for (const auto& item: device_caches_) {
        auto device = item->device;
        if (device && device->clDeviceID && device::get_id(device.get()) == id) {
          auto queue = item->get_next_free_command_queue();
          if (queue) return queue;
        }
      }

      std::shared_ptr<clHelper::Device> next_device = nullptr;
      for (const auto& next: devices_){
        if(next->clDeviceID && device::get_id(next.get()) == id) {
          next_device = next; break;
        }
      }

      cl_command_queue q = nullptr;

      if (next_device) {
        auto item = std::make_shared<gpu_device_item>(next_device);
        device_caches_.push_back(item);
        q = item->get_next_free_command_queue();
      }

      return q;
    }

    void *gpu_device_cache::get_default_command_queue() {
      void* pointer = get_default_device();
      if (!pointer) return nullptr;
      auto* device = reinterpret_cast<clHelper::Device *>(pointer);
      if (device)
        return get_command_queue(device::get_id(device));
      return nullptr;
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

      for (int i = 0; i < (int)kMaxCommandQueues; ++i) {

#ifdef __APPLE__
        auto q = clCreateCommandQueue(context, device_id, 0, &ret);
#elif WIN32
        auto q = clCreateCommandQueue(context, device_id, 0, &ret);
#else
        //cl_queue_properties devQueueProps[] = { 0 };
//        auto q = clCreateCommandQueueWithProperties(context, device_id, devQueueProps, &ret);
        auto q = clCreateCommandQueue(context, device_id, 0, &ret);
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
      #if defined(DEHANCER_OPENCL_CONTEXT_NOT_RELEASE)
      //
      // PS/LR hangs release context
      //
      #else
      if (context)
        clReleaseContext(context);
      #endif
    }

    gpu_command_queue_item::~gpu_command_queue_item() {
      if (command_queue)
        clReleaseCommandQueue(command_queue);
    }

    gpu_command_queue_item::gpu_command_queue_item(bool in_use_, cl_command_queue command_queue_) {
      in_use = in_use_;
      command_queue = command_queue_;
    }

}