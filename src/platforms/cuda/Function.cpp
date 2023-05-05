//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#include "dehancer/gpu/Paths.h"
#include "dehancer/Log.h"
#include "utils.h"

#include <cuda_runtime.h>

namespace dehancer::cuda {
    
    std::mutex Function::mutex_;
    std::unordered_map<CUstream, Function::KernelMap> Function::kernel_map_;
    std::unordered_map<CUstream, Function::ProgamMap> Function::module_map_;
    
    
    void
    Function::execute (CommandEncoder::ComputeSize compute_size, const dehancer::Function::VoidEncodeHandler &block) {
      if (!block) return;
  
      execute_block([block,compute_size](dehancer::CommandEncoder& encoder){
          block(encoder);
          return compute_size;
      });
    }
    
    void Function::execute(const dehancer::Function::EncodeHandler &block) {
  
      execute_block([block](dehancer::CommandEncoder& encoder){
          auto from_block = block(encoder);
          return encoder.ask_compute_size(from_block);
      });
      
    }
    
    void Function::execute_block (const Function::CommonEncodeHandler& block) {
      
      command_->push();
  
      auto encoder = std::make_shared<cuda::CommandEncoder>(kernel_, this);

      auto compute_size = block(*encoder);
      
#ifdef PRINT_KERNELS_DEBUG
      size_t buffer_size = compute_size.threads_in_grid*257*4*sizeof(unsigned int);
      std::cout << "Function " << kernel_name_
                << " global: "
                << compute_size.grid.width << "x" << compute_size.grid.height << "x" << compute_size.grid.depth
                << "  local: "
                << compute_size.block.width << "x" << compute_size.block.height << "x" << compute_size.block.depth
                << "  num_groups: "
                << compute_size.threads_in_grid
                << "  buffer size: "
                <<     buffer_size << "b" << ", " << buffer_size/1024/1204 << "Mb"
                << std::endl;
#endif
  
      cudaEvent_t start, stop;
      if (command_->get_wait_completed()) {
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventCreate(&start));
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventCreate(&stop));
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventRecord(start, nullptr));
      }

#ifdef PRINT_KERNELS_DEBUG
      dehancer::log::print(" === cuda::Function::execute_block[%s] encoder size: %i", kernel_name_.c_str(), encoder->args_.size());
#endif

      CHECK_CUDA_KERNEL(kernel_name_.c_str(),
                        cuLaunchKernel(
                                kernel_,
                                compute_size.grid.width, compute_size.grid.height, compute_size.grid.depth,
                                compute_size.block.width, compute_size.block.height, compute_size.block.depth,
                                0,
                                command_->get_cu_command_queue(),
                                encoder->args_.data(),
                                nullptr)
      );
  
      if (command_->get_wait_completed()) {
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventRecord(stop, nullptr));
        CHECK_CUDA_KERNEL(kernel_name_.c_str(),cudaEventSynchronize(stop));
      }
  
      command_->pop();
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
            max_device_threads_(8)
    {
      
      command_->push();

      max_device_threads_ = command_->get_max_threads();
      
      std::unique_lock<std::mutex> lock(Function::mutex_);
      
      if (kernel_map_.find(command_->get_cu_command_queue()) != kernel_map_.end())
      {
        auto& km =  kernel_map_[command_->get_cu_command_queue()];
        if (km.find(kernel_name_) != km.end()) {
          kernel_ = km[kernel_name_];
          command_->pop();
          return;
        }
      }
      else {
        kernel_map_[command_->get_cu_command_queue()] = {};
      }
      
      CUmodule module = nullptr;
      
      auto p_path = library_path_.empty() ? dehancer::device::get_lib_path() : library_path;
      std::size_t p_path_hash = std::hash<std::string>{}(p_path);
      
      std::string library_source;
      if (p_path.empty()) {
        p_path_hash = dehancer::device::get_lib_source(library_source);
        if (library_source.empty()) {
          command_->pop();
          throw std::runtime_error("Could not find path to CUDA module for '" + kernel_name + "'");
        }
      }
      
      if (module_map_.find(command_->get_cu_command_queue()) != module_map_.end())
      {
        auto& pm =  module_map_[command_->get_cu_command_queue()];
        if (pm.find(p_path_hash) != pm.end()) {
          module = pm[p_path_hash];
        }
      }
      else {
        module_map_[command_->get_cu_command_queue()] = {};
      }
      
      if (module == nullptr) {
        try {
          if (!p_path.empty())
            CHECK_CUDA(cuModuleLoad(&module, p_path.c_str()));
          else {
            CHECK_CUDA(cuModuleLoadData(&module, library_source.data()));
          }
        }
        catch (const std::runtime_error &e) {
          command_->pop();
          throw std::runtime_error(e.what() + std::string(" module: ") + p_path);
        }
        module_map_[command_->get_cu_command_queue()][p_path_hash] = module ;
      }
      
      // Get function handle from module
      try {
        CHECK_CUDA(cuModuleGetFunction(&kernel_, module, kernel_name_.c_str()));
      }
      catch (const std::runtime_error &e) {
        command_->pop();
        throw std::runtime_error(e.what() + std::string(" kernel: ") + kernel_name_);
      }
      
      kernel_map_[command_->get_cu_command_queue()][kernel_name_]=kernel_;
      command_->pop();
    }
    
    Function::~Function() = default;
    
    const std::vector<dehancer::Function::ArgInfo>& Function::get_arg_info_list() const {
      if (arg_list_.empty()) {
        /***
         * TODO: CUDA args info, if it is possible
         */
      }
      return arg_list_;
    }
    
    const std::string &Function::get_name() const {
      return kernel_name_;
    }
    
    const std::string &Function::get_library_path () const {
      return library_path_;
    }
    
    size_t Function::get_block_max_size () const {
      return max_device_threads_;
    }
    
    CommandEncoder::ComputeSize Function::ask_compute_size (size_t width, size_t height, size_t depth) const {
      const auto e = std::make_shared<cuda::CommandEncoder>(kernel_, this);
      return e->ask_compute_size(width, height, depth);
    }
    
  
}
