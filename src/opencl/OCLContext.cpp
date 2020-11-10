//
// Created by denn nevera on 10/11/2020.
//

#include "OCLContext.h"

namespace dehancer::opencl {

    OCLContext::OCLContext(const void *command_queue):command_queue_(command_queue)
    {
      last_error_ = clGetCommandQueueInfo(get_command_queue(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id_,
                                          nullptr);
      if (last_error_ != CL_SUCCESS) {
        throw std::runtime_error("Unable to get OpenCL the device");
      }

      last_error_ = clGetCommandQueueInfo(get_command_queue(), CL_QUEUE_CONTEXT, sizeof(cl_context), &context_,
                                          nullptr);
      if (last_error_ != CL_SUCCESS) {
        throw std::runtime_error("Unable to get OpenCL context");
      }
    }

    cl_command_queue OCLContext::get_command_queue() const {
      return static_cast<cl_command_queue>((void *) command_queue_);
    }

    cl_device_id OCLContext::get_device_id() const {
      return device_id_;
    }

    cl_context OCLContext::get_context() const {
      return context_;
    }
}