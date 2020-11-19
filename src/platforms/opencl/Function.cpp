//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#include "dehancer/gpu/Paths.h"

namespace dehancer::opencl {

    void Function::execute(const dehancer::Function::FunctionHandler &block) {

      auto texture = block(*encoder_);

      if (!texture) return;

      auto device_id = command_->get_device_id();

      size_t localWorkSize[2] = {1,1};

      clGetKernelWorkGroupInfo(kernel_,  device_id,
                               CL_KERNEL_WORK_GROUP_SIZE, sizeof(localWorkSize), localWorkSize, nullptr);

      if (localWorkSize[0]>=texture->get_width()) localWorkSize[0] = texture->get_width();
      if (localWorkSize[1]>=texture->get_height()) localWorkSize[1] = texture->get_height();

      size_t globalWorkSize[2] = {
              ((texture->get_width() + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0],
              ((texture->get_height() + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1]
      };

      cl_int last_error = 0;
      cl_event    waiting_event = nullptr;

      if (command_->get_wait_completed())
        waiting_event = clCreateUserEvent(command_->get_context(), &last_error);

      last_error = clEnqueueNDRangeKernel(command_->get_command_queue(), kernel_, 2, nullptr,
                                          globalWorkSize,
                                          localWorkSize,
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

    Function::Function(dehancer::opencl::Command *command, const std::string& kernel_name):
            command_(command),
            kernel_name_(kernel_name),
            program_(nullptr),
            kernel_(nullptr),
            encoder_(nullptr),
            arg_list_({})
    {
      const std::string source = clHelper::getEmbeddedProgram(dehancer::device::get_lib_path());

      const char *source_str = source.c_str();
      size_t source_size = source.size();

      cl_int  last_error;

      program_ = clCreateProgramWithSource(command_->get_context(), 1, (const char **) &source_str,
                                           (const size_t *) &source_size, &last_error);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
      }

      /* Build Kernel Program */
      auto device_id = command_->get_device_id();
      last_error = clBuildProgram(program_, 1, &device_id, "-cl-kernel-arg-info", nullptr, nullptr);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to build OpenCL program from exampleKernel.cl");
      }

      kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &last_error);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to create kernel for: " + kernel_name_);
      }

      encoder_ = std::make_shared<opencl::CommandEncoder>(kernel_);
    }

    Function::~Function() {
      if (kernel_)
        clReleaseKernel(kernel_);
      if (program_)
        clReleaseProgram(program_);
    }

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
