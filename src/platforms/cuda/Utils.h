//
// Created by denn on 28.12.2020.
//

#pragma once

#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace dehancer::cuda {
    template <typename T>
    static inline void check(T result, char const *const func, const char *const file, int const line) {
      if (result) {
        std::stringstream ss;
        ss << "CUDA error at " << file << ":" << line;
        ss << " " << static_cast<unsigned int>(result) << "("  << cudaGetErrorName(static_cast<cudaError_t>(result)) << ")";
        ss << " " << func;
        throw std::runtime_error(ss.str());
      }
    }
}

#define CHECK_CUDA(val) dehancer::cuda::check((val), #val, __FILE__, __LINE__)
