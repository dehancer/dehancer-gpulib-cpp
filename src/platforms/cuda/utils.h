//
// Created by denn on 28.12.2020.
//

#pragma once

#ifndef CUDA_KERNEL

#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace dehancer::cuda {
    template<typename T>
    static inline void check(T result, char const *const func, const char *const file, int const line, const char* kname) {
      if (result) {
        std::stringstream ss;
        ss << "CUDA error at " << file << ":" << line;
        ss << " " << static_cast<unsigned int>(result) << "(" << cudaGetErrorName(static_cast<::cudaError_t>(result))
           << "), kernel: " << kname << "; ";
        ss << ": " << func;
        throw std::runtime_error(ss.str());
      }
    }
}


#define CHECK_CUDA(val) dehancer::cuda::check((val), #val, __FILE__, __LINE__, "")
#define CHECK_CUDA_KERNEL(kname,val) dehancer::cuda::check((val), #val, __FILE__, __LINE__, kname)

#endif