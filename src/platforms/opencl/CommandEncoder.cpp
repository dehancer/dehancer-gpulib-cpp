//
// Created by denn nevera on 10/11/2020.
//

#include "CommandEncoder.h"

namespace dehancer::opencl {

    CommandEncoder::CommandEncoder(cl_kernel kernel): kernel_(kernel){}

    void CommandEncoder::set(const Texture &texture, int index)  {
      if (kernel_) {
        auto memobj = static_cast<cl_mem>(texture->get_memory());
        auto ret = clSetKernelArg(kernel_, index, sizeof(cl_mem), (void *)&memobj);
        if (ret != CL_SUCCESS)
          throw std::runtime_error("Unable to pass to kernel the texture buffer at index: " + std::to_string(index));
      }
      else {
        throw std::runtime_error("Unable to pass texture to null kernel ");
      }
    }

    void CommandEncoder::set(const void *bytes, size_t bytes_length, int index)  {
      if (kernel_){
        auto ret = clSetKernelArg(kernel_, 0, bytes_length,  bytes);
        if (ret != CL_SUCCESS)
          throw std::runtime_error("Unable to pass to kernel bytes at index: " + std::to_string(index));
      }
      else {
        throw std::runtime_error("Unable to pass bytes to null kernel ");
      }
    }
}