//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#include "dehancer/gpu/Paths.h"
#include "dehancer/gpu/Log.h"
#include "OpenCLCache.h"

namespace dehancer::opencl {
    
    std::mutex Function::mutex_;
    std::unordered_map<cl_command_queue, Function::KernelMap> Function::kernel_map_;
    std::unordered_map<cl_command_queue, Function::ProgramMap> Function::program_map_;
    
    size_t upper_power2(size_t x) {
      union { double f; long i[2]; } convert{};
      convert.f = static_cast<double>(x);
      if ((convert.i[1] & 0xFFFFF) | convert.i[0]) return 1<<((convert.i[1]>>20) - 0x3FE);
      else return x;
    }
    
    void Function::execute(CommandEncoder::ComputeSize compute_size,
                           const dehancer::Function::VoidEncodeHandler& block) {
  
      std::unique_lock<std::mutex> lock(Function::mutex_);
      
      if (block)
        block(*encoder_);
      
      cl_int last_error = 0;
      cl_event    waiting_event = nullptr;
  
      cl_uint dim = 3;
  
      if ( (compute_size.grid.depth == 1) && (compute_size.block.depth == 1) ) {
        dim = 2;
        if ((compute_size.grid.height == 1) && (compute_size.block.height == 1)) dim = 1;
      }
      
      if (command_->get_wait_completed())
        waiting_event = clCreateUserEvent(command_->get_context(), &last_error);
  
      size_t global_work_size[3] = {compute_size.grid.width, compute_size.grid.height, compute_size.grid.depth };
      size_t local_work_size[3]  = {compute_size.block.width,compute_size.block.height,compute_size.block.depth};
  
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

      last_error = clEnqueueNDRangeKernel(command_->get_cl_command_queue(),
                                          kernel_,
                                          dim,
                                          nullptr,
                                          global_work_size,
                                          local_work_size,
                                          0,
                                          nullptr,
                                          &waiting_event);


      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to enqueue kernel: " + kernel_name_ + " error code: " + std::to_string(last_error));
      }
  
      if (waiting_event && command_->get_wait_completed()) {

        last_error = clWaitForEvents(1, &waiting_event);
    
        clReleaseEvent(waiting_event);
    
        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to waiting execution of kernel: " + kernel_name_ + " error code: " + std::to_string(last_error));
        }
      }
    }
    
    void Function::execute(const dehancer::Function::EncodeHandler &block) {
  
      CommandEncoder::Size texture_size{};
      
      {
        std::unique_lock<std::mutex> lock(Function::mutex_);
        
        texture_size = block(*encoder_);
      }
      
      auto compute_size = encoder_->ask_compute_size(texture_size);
      
      #ifdef PRINT_KERNELS_DEBUG
      std::cout << "Function " << kernel_name_
                << " CL_KERNEL_WORK_GROUP_SIZE: "
                << encoder_->get_block_max_size()
                << std::endl;
      #endif
      
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
      
      execute(compute_size, nullptr);
    }

    Function::Function(
            dehancer::opencl::Command *command,
            const std::string &kernel_name,
            const std::string &library_path) :
            command_(command),
            kernel_name_(kernel_name),
            library_path_(library_path),
            kernel_(nullptr),
            encoder_(nullptr),
            arg_list_({}) {
        std::unique_lock<std::mutex> lock(Function::mutex_);

        if (kernel_map_.find(command_->get_cl_command_queue()) != kernel_map_.end()) {
            auto &km = kernel_map_[command_->get_cl_command_queue()];
            if (km.find(kernel_name_) != km.end()) {
                kernel_ = km[kernel_name_];
                encoder_ = std::make_shared<opencl::CommandEncoder>(kernel_, this);
                return;
            }
        } else {
            kernel_map_[command_->get_cl_command_queue()] = {};
        }

        cl_program program_ = nullptr;

        auto p_path = library_path_.empty() ? dehancer::device::get_lib_path() : library_path;
        std::size_t p_path_hash = std::hash<std::string>{}(p_path);

        std::string library_source_;
        if (p_path.empty()) {
            p_path_hash = dehancer::device::get_lib_source(library_source_);
            if (library_source_.empty())
                throw std::runtime_error("Could not find embedded opencl source code for '" + kernel_name + "'");
        }

        if (program_map_.find(command_->get_cl_command_queue()) != program_map_.end()) {
            auto &pm = program_map_[command_->get_cl_command_queue()];
            if (pm.find(p_path_hash) != pm.end()) {
                program_ = pm[p_path_hash];
            }
        } else {
            program_map_[command_->get_cl_command_queue()] = {};
        }

        cl_int last_error = 0;

        if (program_ == nullptr) {
            std::string source;
            const char *source_str;
            size_t source_size = source.size();

            if (library_source_.empty()) {
                source = clHelper::getEmbeddedProgram(p_path);
                source_str = source.c_str();
                source_size = source.size();
            } else {
                source = library_source_;
                source_str = library_source_.c_str();
                source_size = library_source_.size();
            }

            program_ = OpenCLCache::Instance().program_for_source(command_->get_context(), source,
                                                                  command_->get_device_id());

//            program_ = clCreateProgramWithSource(command_->get_context(), 1, (const char **) &source_str,
//                                                 (const size_t *) &source_size, &last_error);
//
//            if (last_error != CL_SUCCESS) {
//                throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
//            }

            /* Build Kernel Program */
            auto device_id = command_->get_device_id();
            last_error = clBuildProgram(program_, 1, &device_id,
                                        "-cl-std=CL2.0 -cl-kernel-arg-info -cl-unsafe-math-optimizations -cl-single-precision-constant",
                                        nullptr, nullptr);

            if (last_error != CL_SUCCESS) {

                std::string log = "Unable to build OpenCL program from: " + kernel_name_;

                // Determine the size of the log
                size_t log_size;
                clGetProgramBuildInfo(program_, command_->get_device_id(), CL_PROGRAM_BUILD_LOG,
                                      0, nullptr, &log_size);
                log.resize(log_size);

                // Get the log
                clGetProgramBuildInfo(program_, command_->get_device_id(), CL_PROGRAM_BUILD_LOG,
                                      log_size, log.data(), nullptr);

                log::error(true, "OpenCL Function build Error[%i]: %s", last_error, log.c_str());
                throw std::runtime_error(
                        "Unable to build OpenCL program from: '" + p_path + "' on: " + kernel_name_ + ": \n[" +
                        std::to_string(log_size) + "] " + log);
            }

            program_map_[command_->get_cl_command_queue()][p_path_hash] = program_;
        }

        kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &last_error);

        kernel_map_[command_->get_cl_command_queue()][kernel_name_] = kernel_;

        if (last_error != CL_SUCCESS) {
            throw std::runtime_error("Unable to create kernel for: " + kernel_name_);
        }

        encoder_ = std::make_shared<opencl::CommandEncoder>(kernel_, this);
    }
    
    Function::~Function() = default;
    
    const std::vector<dehancer::Function::ArgInfo>& Function::get_arg_info_list() const {
      if (arg_list_.empty()) {
        cl_int arg_num = 0;
        cl_int last_error = clGetKernelInfo (	kernel_,
                                                 CL_KERNEL_NUM_ARGS,
                                                 sizeof(cl_uint),
                                                 &arg_num,
                                                 nullptr
        );
        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to get kernel info for: " + kernel_name_);
        }
        for (int i = 0; i < arg_num; ++i) {
          char name[256];
          char type_name[256];
          last_error = clGetKernelArgInfo(
                  kernel_,
                  i,
                  CL_KERNEL_ARG_NAME,
                  sizeof(name),
                  name,
                  nullptr);
          if (last_error != CL_SUCCESS) {
            throw std::runtime_error("Unable to get kernel arg info for: " + kernel_name_);
          }
          last_error = clGetKernelArgInfo(
                  kernel_,
                  i,
                  CL_KERNEL_ARG_TYPE_NAME,
                  sizeof(type_name),
                  type_name,
                  nullptr);
          if (last_error != CL_SUCCESS) {
            throw std::runtime_error("Unable to get kernel arg info for: " + kernel_name_);
          }
          dehancer::Function::ArgInfo info = {
                  .name = name,
                  .index = static_cast<uint>(i),
                  .type_name = type_name
          };
          
          arg_list_.push_back(info);
        }
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
      return encoder_->get_block_max_size();
    }
    
    CommandEncoder::ComputeSize Function::ask_compute_size (size_t width, size_t height, size_t depth) const {
      return encoder_->ask_compute_size(width,height,depth);
    }
}
