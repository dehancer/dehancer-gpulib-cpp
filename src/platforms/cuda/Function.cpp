//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#include "dehancer/gpu/Paths.h"
#include "utils.h"

#include <cuda_runtime.h>

namespace dehancer::cuda {

    std::mutex Function::mutex_;
    std::unordered_map<CUstream, Function::KernelMap> Function::kernel_map_;
    std::unordered_map<CUstream, Function::ProgamMap> Function::module_map_;

    void Function::execute(const dehancer::Function::EncodeHandler &block) {

      std::unique_lock<std::mutex> lock(Function::mutex_);

      auto encoder = std::make_shared<cuda::CommandEncoder>(kernel_, this);

      auto texture_size = block(*encoder);

      float pow_coef = 3 ;

      if (texture_size.depth==1) {
        pow_coef = 2;
      }

      if (texture_size.height==1) {
        pow_coef = 1;
      }

      auto size = (int)((pow(max_device_threads_,1/pow_coef) - 1)/2);
      size |= size >> 1; size |= size >> 2; size |= size >> 4; size |= size >> 8; size |= size >> 16; size += 1;

      dim3 block_size(size, size, size);

      if (texture_size.depth==1) {
        block_size.z = 1;
        block_size.x *=2;
      }

      if (texture_size.height==1) {
        block_size.y = 1;
      }

      if (texture_size.width < block_size.x) block_size.x = texture_size.width;
      if (texture_size.height < block_size.y) block_size.y = texture_size.height;
      if (texture_size.depth < block_size.z) block_size.z = texture_size.depth;

      dim3 grid_size((texture_size.width  + block_size.x - 1) / block_size.x,
                     (texture_size.height + block_size.y - 1) / block_size.y,
                     (texture_size.depth + block_size.z - 1) / block_size.z
      );

#ifdef PRINT_KERNELS_DEBUG
      std::cout << "Function "<<kernel_name_<<"  max threads: "
                << max_device_threads_
                << " blocks: "
                << block_size.x << "x" << block_size.y << "x" << block_size.z
                << " grid: "
                << grid_size.x << "x" << grid_size.y << "x" << grid_size.z
                << std::endl;
#endif

      cudaEvent_t start, stop;
      if (command_->get_wait_completed()) {
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventCreate(&start));
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventCreate(&stop));
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventRecord(start, nullptr));
      }

      CHECK_CUDA_KERNEL(kernel_name_.c_str(),cuLaunchKernel(
              kernel_,
              grid_size.x, grid_size.y, grid_size.z,
              block_size.x, block_size.y, block_size.z,
              0,
              command_->get_command_queue(),
              encoder->args_.data(),
              nullptr)
      );

      if (command_->get_wait_completed()) {
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventRecord(stop, nullptr));
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventSynchronize(stop));
      }
    }

    Function::Function(
            dehancer::cuda::Command *command,
            const std::string& kernel_name,
            const std::string &library_path):
            command_(command),
            kernel_name_(kernel_name),
            library_path_(library_path),
            kernel_(nullptr),
            arg_list_({}),
            function_context_(nullptr),
            current_context_(nullptr),
            max_device_threads_(8)
    {
      std::unique_lock<std::mutex> lock(Function::mutex_);

      CHECK_CUDA(cuCtxGetCurrent(&current_context_));

      CHECK_CUDA(cuStreamGetCtx(command_->get_command_queue(), &function_context_));

      if (function_context_ != current_context_) {

        CHECK_CUDA(cuCtxPopCurrent(&current_context_));

        CHECK_CUDA(cuCtxPushCurrent(function_context_));
      }

      CUdevice cUdevice_0 = -1;
      CHECK_CUDA(cuCtxGetDevice(&cUdevice_0));

      cudaDeviceProp props{};

      cudaGetDeviceProperties(&props, cUdevice_0);
  
      if (function_context_ != current_context_) {
        CHECK_CUDA(cuCtxPopCurrent(&function_context_));
        CHECK_CUDA(cuCtxPushCurrent( current_context_ ));
      }
      
      max_device_threads_ = props.maxThreadsPerBlock;

      if (kernel_map_.find(command_->get_command_queue()) != kernel_map_.end())
      {
        auto& km =  kernel_map_[command_->get_command_queue()];
        if (km.find(kernel_name_) != km.end()) {
          kernel_ = km[kernel_name_];
          return;
        }
      }
      else {
        kernel_map_[command_->get_command_queue()] = {};
      }

      CUmodule module = nullptr;

      auto p_path = library_path_.empty() ? dehancer::device::get_lib_path() : library_path;
      std::size_t p_path_hash = std::hash<std::string>{}(p_path);

      if (p_path.empty()) {
        throw std::runtime_error("Could not find path to CUDA module for '" + kernel_name + "'");
      }

      if (module_map_.find(command_->get_command_queue()) != module_map_.end())
      {
        auto& pm =  module_map_[command_->get_command_queue()];
        if (pm.find(p_path_hash) != pm.end()) {
          module = pm[p_path_hash];
        }
      }
      else {
        module_map_[command_->get_command_queue()] = {};
      }

      if (module == nullptr) {
        try {
          CHECK_CUDA(cuModuleLoad(&module, p_path.c_str()));
        }
        catch (const std::runtime_error &e) {
          throw std::runtime_error(e.what() + std::string(" module: ") + p_path);
        }
        module_map_[command_->get_command_queue()][p_path_hash] = module ;
      }

      // Get function handle from module
      try {
        CHECK_CUDA(cuModuleGetFunction(&kernel_, module, kernel_name_.c_str()));
      }
      catch (const std::runtime_error &e) {
        throw std::runtime_error(e.what() + std::string(" kernel: ") + kernel_name_);
      }

      kernel_map_[command_->get_command_queue()][kernel_name_]=kernel_;
    }

    Function::~Function() {
//      if (function_context_ != current_context_) {
//        CHECK_CUDA(cuCtxPopCurrent(&function_context_));
//        CHECK_CUDA(cuCtxPushCurrent( current_context_ ));
//      }
    }

    const std::vector<dehancer::Function::ArgInfo>& Function::get_arg_info_list() const {
      if (arg_list_.empty()) {

      }
      return arg_list_;
    }

    const std::string &Function::get_name() const {
      return kernel_name_;
    }
}
