//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#include "dehancer/gpu/Paths.h"

namespace dehancer::opencl {

    std::mutex Function::mutex_;
    std::unordered_map<cl_command_queue, Function::KernelMap> Function::kernel_map_;
    std::unordered_map<cl_command_queue, Function::ProgamMap> Function::program_map_;

    void Function::execute(const dehancer::Function::FunctionHandler &block) {

      std::unique_lock<std::mutex> lock(Function::mutex_);

      auto texture_size = block(*encoder_);

      auto device_id = command_->get_device_id();


      size_t local_work_size[2];

      clGetKernelWorkGroupInfo(kernel_, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), local_work_size, nullptr);

      local_work_size[1] = 1;

      if (local_work_size[0]>=texture_size.width) local_work_size[0] = 1;

      size_t global_work_size[2] = {
              ((texture_size.width + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0],
              texture_size.height
      };

      cl_int last_error = 0;
      cl_event    waiting_event = nullptr;

      if (command_->get_wait_completed())
        waiting_event = clCreateUserEvent(command_->get_context(), &last_error);

      last_error = clEnqueueNDRangeKernel(command_->get_command_queue(), kernel_, 2, nullptr,
                                          global_work_size,
                                          local_work_size,
                                          0,
                                          nullptr,
                                          &waiting_event);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to enqueue kernel: " + kernel_name_);
      }

      if (waiting_event && command_->get_wait_completed()) {
        last_error = clWaitForEvents(1, &waiting_event);

        clReleaseEvent(waiting_event);

        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to waiting execution of kernel: " + kernel_name_);
        }
      }
    }

    Function::Function(
            dehancer::opencl::Command *command,
            const std::string& kernel_name,
            const std::string &library_path):
            command_(command),
            kernel_name_(kernel_name),
            library_path_(library_path),
            kernel_(nullptr),
            encoder_(nullptr),
            arg_list_({})
    {
      std::unique_lock<std::mutex> lock(Function::mutex_);

      if (kernel_map_.find(command_->get_command_queue()) != kernel_map_.end())
      {
        auto& km =  kernel_map_[command_->get_command_queue()];
        if (km.find(kernel_name_) != km.end()) {
          kernel_ = km[kernel_name_];
          encoder_ = std::make_shared<opencl::CommandEncoder>(kernel_,this);
          return;
        }
      }
      else {
        kernel_map_[command_->get_command_queue()] = {};
      }

      cl_program program_ = nullptr;

      auto p_path = library_path_.empty() ? dehancer::device::get_lib_path() : library_path;

      if (program_map_.find(command_->get_command_queue()) != program_map_.end())
      {
        auto& pm =  program_map_[command_->get_command_queue()];
        if (pm.find(p_path) != pm.end()) {
          program_ = pm[p_path];
        }
      }
      else {
        program_map_[command_->get_command_queue()] = {};
      }

      cl_int last_error = 0;

      if (program_ == nullptr) {
        const std::string source = clHelper::getEmbeddedProgram(p_path);

         const char *source_str = source.c_str();
        size_t source_size = source.size();


        program_ = clCreateProgramWithSource(command_->get_context(), 1, (const char **) &source_str,
                                                        (const size_t *) &source_size, &last_error);

        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
        }

        /* Build Kernel Program */
        auto device_id = command_->get_device_id();
        last_error = clBuildProgram(program_, 1, &device_id, "-cl-kernel-arg-info", nullptr, nullptr);

        if (last_error != CL_SUCCESS) {

          std::string log = "Unable to build OpenCL program from: " + kernel_name_;

          if (last_error == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program_, command_->get_device_id(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            //build_log_.resize(log_size);
            log.resize(log_size);
            // Get the log
            clGetProgramBuildInfo(program_, command_->get_device_id(), CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                                  NULL);
          }

          throw std::runtime_error("Unable to build OpenCL program from: " + kernel_name_ + ": \n" + log);
        }

        program_map_[command_->get_command_queue()][p_path] = program_ ;
      }

      kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &last_error);

      kernel_map_[command_->get_command_queue()][kernel_name_]=kernel_;

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
}
