//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "OCLContext.h"

namespace dehancer::opencl {

    class OCLCommandEncoder: public CommandEncoder {

    public:
        explicit OCLCommandEncoder(cl_kernel kernel):kernel_(kernel){}

        void set(const Texture &texture, int index) override {
          if (kernel_) {
            auto memobj = static_cast<cl_mem>(texture->get_contents());
            auto ret = clSetKernelArg(kernel_, index, sizeof(cl_mem), (void *)&memobj);
            if (ret != CL_SUCCESS) throw std::runtime_error("Unable to pass to kernel the source texture buffer");
          }
          else {
            throw std::runtime_error("Unable to pass texture to null kernel ");
          }
        }

        void set(const void *bytes, size_t bytes_length, int index) override {
          if (kernel_){
            auto ret = clSetKernelArg(kernel_, 0, bytes_length,  bytes);
            if (ret != CL_SUCCESS) throw std::runtime_error("Unable to pass to kernel bytes");
          }
          else {
            throw std::runtime_error("Unable to pass bytes to null kernel ");
          }
        }

        cl_kernel kernel_ = nullptr;
    };
}
