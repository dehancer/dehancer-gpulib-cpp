//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#include "dehancer/gpu/Paths.h"
#include "dehancer/gpu/kernels/cuda/utils.h"

#include <cuda_runtime.h>

namespace dehancer::cuda {

    std::mutex Function::mutex_;
    std::unordered_map<CUstream, Function::KernelMap> Function::kernel_map_;
    std::unordered_map<CUstream, Function::ProgamMap> Function::module_map_;

    void Function::execute(const dehancer::Function::FunctionHandler &block) {

      std::unique_lock<std::mutex> lock(Function::mutex_);

      auto encoder = std::make_shared<cuda::CommandEncoder>(kernel_, this);

      auto texture_size = block(*encoder);

      ///
      /// TODO: optimize workgroups automatically
      ///

      // int block_size;      // The launch configurator returned block size
      // int min_grid_size = std::max(texture_size.width,texture_size.height);   // The minimum grid size needed to achieve the maximum occupancy for a full device launch
      // int grid_size;       // The actual grid size needed, based on input size
      // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_, 0, min_grid_size);

      //cudaFuncAttributes attr{};
      //CHECK_CUDA(cudaFuncGetAttributes (&attr, kernel_));

      //std::cout << " Function attr: maxThreadsPerBlock: " << attr.maxThreadsPerBlock << std::endl;

      // Check stream
      CUcontext cuContext_0, current_context;

      CHECK_CUDA(cuCtxGetCurrent(&current_context));

      CHECK_CUDA(cuStreamGetCtx(command_->get_command_queue(), &cuContext_0));

      CHECK_CUDA(cuCtxPopCurrent(&current_context));

      CHECK_CUDA(cuCtxPushCurrent(cuContext_0));

      CUdevice cUdevice_0 = -1;
      CHECK_CUDA(cuCtxGetDevice(&cUdevice_0));

      cudaDeviceProp props{};

      cudaGetDeviceProperties(&props, cUdevice_0);

      float pow_coef = 3 ;

      if (texture_size.depth==1) {
        pow_coef = 2;
      }

      if (texture_size.height==1) {
        pow_coef = 1;
      }

      auto size = (int)((pow(props.maxThreadsPerBlock,1/pow_coef) - 1)/2);
      size |= size >> 1; size |= size >> 2; size |= size >> 4; size |= size >> 8; size |= size >> 16; size += 1;

      std::cout << " Function attr: maxThreadsPerBlock: " << props.maxThreadsPerBlock << " blocks: " << size << std::endl;

      dim3 block_size(size, size, size);

      if (texture_size.depth==1) {
        block_size.z = 1;
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

      cudaEvent_t start, stop;
      if (command_->get_wait_completed()) {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, nullptr));
      }

      CHECK_CUDA(cuLaunchKernel(
              kernel_,
              grid_size.x, grid_size.y, grid_size.z,
              block_size.x, block_size.y, block_size.z,
              0,
              command_->get_command_queue(),
              encoder->args_.data(),
              nullptr)
      );

      if (command_->get_wait_completed()) {
        CHECK_CUDA(cudaEventRecord(stop, nullptr));
        CHECK_CUDA(cudaEventSynchronize(stop));
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
            arg_list_({})
    {
      std::unique_lock<std::mutex> lock(Function::mutex_);

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

      std::string library_source_;
      if (p_path.empty()) {
        p_path_hash = dehancer::device::get_lib_source(library_source_);
        if (library_source_.empty())
          throw std::runtime_error("Could not find embedded cuda source code for '" + kernel_name + "'");
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
        std::string source;
        const char *source_str;
        size_t source_size = source.size();

        if (!library_source_.empty()) {
          //source = clHelper::getEmbeddedProgram(p_path);
          //source_str = source.c_str();
          //source_size = source.size();
          assert(0);
        }
        else {
          //source_str = library_source_.c_str();
          //source_size = library_source_.size();
        }


        std::cout << "Module: " << dehancer::device::get_lib_path() << std::endl;
        CHECK_CUDA(cuModuleLoad(&module, p_path.c_str()));

        module_map_[command_->get_command_queue()][p_path_hash] = module ;
      }

      // Get function handle from module
      CHECK_CUDA(cuModuleGetFunction(&kernel_, module, kernel_name_.c_str()));

      //kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &last_error);

      kernel_map_[command_->get_command_queue()][kernel_name_]=kernel_;

//      if (last_error != CL_SUCCESS) {
//        throw std::runtime_error("Unable to create kernel for: " + kernel_name_);
//      }
    }

    Function::~Function() = default;

    const std::vector<dehancer::Function::ArgInfo>& Function::get_arg_info_list() const {
      if (arg_list_.empty()) {

      }
      return arg_list_;
    }

    const std::string &Function::get_name() const {
      return kernel_name_;
    }
}
