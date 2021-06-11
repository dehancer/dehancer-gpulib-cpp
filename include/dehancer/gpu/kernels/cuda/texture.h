//
// Created by denn on 30.12.2020.
//

#pragma once

#ifndef DEHANCER_CUDA_TEXTURE_KERNELS_H
#define DEHANCER_CUDA_TEXTURE_KERNELS_H

#include <cuda.h>

namespace dehancer {
    
    namespace nvcc {
        
        struct texture {
#ifndef CUDA_KERNEL
            __host__ [[nodiscard]] virtual const cudaArray *get_contents() const = 0;
            __host__ [[nodiscard]] virtual cudaArray *get_contents() = 0;
#endif
            __device__ [[nodiscard]] virtual size_t get_width() const = 0;
            __device__ [[nodiscard]] virtual size_t get_height() const = 0;
            __device__ [[nodiscard]] virtual size_t get_depth() const = 0 ;
        };
        
    }
}

#endif