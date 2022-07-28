//
// Created by denn nevera on 15/11/2020.
//

#include "DeviceCache.h"

namespace dehancer::cuda {
    
    namespace device {
        std::string get_name(const void* id) {
          auto* item = static_cast<const gpu_device_item*>(id);
          if (!item)
            return "unknown";
          return item->device->props.name;
        }
        
        uint64_t    get_id(const void* id) {
          auto* item = static_cast<const gpu_device_item*>(id);
          if (!item)
            return UINT64_MAX;
          return item->device->device_id;
        }
        
        dehancer::device::Type get_type(const void* id){
          return dehancer::device::Type::gpu;
        }
    }
    
    device_helper::device_helper(CUdevice id) {
      device_id = id;
      cudaGetDeviceProperties(&props, device_id);
    }
    
    gpu_device_cache::gpu_device_cache():
            device_caches_(),
            default_device_index_(0)
    {
      int numDevices = 0;
      int maxMultiprocessors = 0;
      cudaGetDeviceCount(&numDevices);
      for (int i=0; i<numDevices; ++i){
        auto d = std::make_shared<gpu_device_item>(i);
        device_caches_.push_back(d);
        if(maxMultiprocessors < d->device->props.multiProcessorCount) {
          maxMultiprocessors = d->device->props.multiProcessorCount;
          default_device_index_ = i;
        }
      }
    }
    
    std::vector<void *> gpu_device_cache::get_device_list(dehancer::device::TypeFilter filter) {
      std::vector<void *> list;
      for(const auto& d: device_caches_){
        #ifdef DEHANCER_GPU_CUDA_CACHE_DEBUG
        auto* item = static_cast<const gpu_device_item*>(d.get());
        std::cout << "                           Device: " << item->device->props.name << std::endl
                  << "                            major: " << item->device->props.major << std::endl
                  << "                            minor: " << item->device->props.minor << std::endl
                  << "                         memPitch: " << item->device->props.memPitch << std::endl
                  << "                      computeMode: " << item->device->props.computeMode << std::endl
                  << "                 textureAlignment: " << item->device->props.textureAlignment << std::endl
                  << "            texturePitchAlignment: " << item->device->props.texturePitchAlignment << std::endl
                  << "                 surfaceAlignment: " << item->device->props.surfaceAlignment << std::endl
                  << "                       ECCEnabled: " << item->device->props.ECCEnabled << std::endl
                  << "                unifiedAddressing: " << item->device->props.unifiedAddressing << std::endl
                  << " singleToDoublePrecisionPerfRatio: " << item->device->props.singleToDoublePrecisionPerfRatio << std::endl
                  << "   directManagedMemAccessFromHost: " << item->device->props.directManagedMemAccessFromHost << std::endl
                  << " === " << std::endl
                  << std::endl;
        #endif
        if (filter & device::get_type(d.get()))
          list.push_back(d.get());
      }
      return list;
    }
    
    void *gpu_device_cache::get_device(uint64_t id) {
      
      for (const auto& item: device_caches_) {
        auto device = item->device;
        if (device && device->device_id>=0 && device::get_id(item.get()) == id) {
          return item->device.get();
        }
      }
      return nullptr;
    }
    
    void *gpu_device_cache::get_default_device() {
      if (!device_caches_.empty()) {
        auto it = device_caches_[default_device_index_];
        return it.get();
      }
      return nullptr;
    }
    
    void *gpu_device_cache::get_command_queue(uint64_t id) {
      
      for (const auto& item: device_caches_) {
        const auto& device = item;
        if (device && device->device->device_id>=0 && device::get_id(device.get()) == id) {
          auto queue = item->get_next_free_command_queue();
          if (queue) return queue;
        }
      }
      
      std::shared_ptr<gpu_device_item> next_device = nullptr;
      for (const auto& next: device_caches_){
        if(next && next->device->device_id>=0 && device::get_id(next.get()) == id) {
          next_device = next; break;
        }
      }
      
      cudaStream_t q = nullptr;
      
      if (next_device) {
        auto item = std::make_shared<gpu_device_item>(next_device->device->device_id);
        device_caches_.push_back(item);
        q = item->get_next_free_command_queue();
      }
      
      return q;
    }
    
    void *gpu_device_cache::get_default_command_queue() {
      auto d = get_default_device();
      if (!d) return nullptr;
      auto* device = reinterpret_cast<gpu_device_item*>(d);
      if (device)
        return get_command_queue(device::get_id(device));
      return nullptr;
    }
    
    void gpu_device_cache::return_command_queue(const void *q) {
      for (auto& device: device_caches_) {
        if(device->return_command_queue(reinterpret_cast<cudaStream_t>((void*)q)))
          break;
      }
    }
    
    gpu_device_item::gpu_device_item(const CUdevice& id)
            :device(std::make_shared<device_helper>(id))
    {
      CUdevice device_id = device->device_id;
      
      CHECK_CUDA(cuCtxCreate(&context, 0 , device_id));
      
      CHECK_CUDA(cuCtxPushCurrent(context));
      
      for (size_t i = 0; i < kMaxCommandQueues; ++i) {
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        command_queue_cache.push_back(std::make_shared<gpu_command_queue_item>(false,stream));
      }
      
      CHECK_CUDA(cuCtxPopCurrent(&context));
    }
    
    cudaStream_t gpu_device_item::get_next_free_command_queue() {
      std::lock_guard lock(mutex_);
      for(auto& item: command_queue_cache) {
        if (item->in_use) continue;
        item->in_use = true;
        return item->command_queue;
      }
      return nullptr;
    }
    
    bool gpu_device_item::return_command_queue(cudaStream_t command_queue) {
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
      for (auto & i : command_queue_cache) {
        CHECK_CUDA(cuStreamDestroy(i->command_queue));
      }
      if (context) {
        CHECK_CUDA(cuCtxDestroy(context));
      }
      context = nullptr;
    }
    
    gpu_command_queue_item::gpu_command_queue_item(bool in_use_, cudaStream_t command_queue_) {
      in_use = in_use_;
      command_queue = command_queue_;
    }
}