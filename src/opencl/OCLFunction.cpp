//
// Created by denn nevera on 10/11/2020.
//

#include "OCLFunction.h"
#include "OCLCommandEncoder.h"

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

      cl_event    AlphaComposting12 = nullptr;

      auto last_error = clEnqueueNDRangeKernel(command_->get_command_queue(), kernel_, 2, nullptr,
                                           globalWorkSize,
                                           localWorkSize,
                                           0,
                                           nullptr, &AlphaComposting12);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to enqueue kernel: " + kernel_name_);
      }

      if (command_->get_wait_completed()) {
        last_error = clWaitForEvents(1, &AlphaComposting12);

        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to enqueue kernel: " + kernel_name_);
        }
      }
    }

    Function::Function(dehancer::opencl::Command *command, const std::string& kernel_name):
            command_(command),
            kernel_name_(kernel_name),
            program_(nullptr),
            kernel_(nullptr),
            encoder_(nullptr)
    {
      const std::string source = clHelper::getEmbeddedProgram("exampleKernel.cl");

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
      last_error = clBuildProgram(program_, 1, &device_id, nullptr, nullptr, nullptr);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to build OpenCL program from exampleKernel.cl");
      }

      kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &last_error);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to create kernel for: " + kernel_name_);
      }

      encoder_ = std::make_shared<opencl::OCLCommandEncoder>(kernel_);
    }

    Function::~Function() {
      if (kernel_)
        clReleaseKernel(kernel_);
      if (program_)
        clReleaseProgram(program_);
    }
}
