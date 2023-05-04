//
// Created by denn on 30.12.2020.
//

#pragma once

#ifndef DEHANCER_CUDA_TEXTURE_KERNELS_H
#define DEHANCER_CUDA_TEXTURE_KERNELS_H

#include <cuda.h>
#include <cuda_fp16.h>


#define _HALF_FLOAT_SIZE_BASE_ (65535)
#define _HALF_FLOAT_SIZE_      (_HALF_FLOAT_SIZE_BASE_>>1)
#define _HALF_FLOAT_SIZE_MAX_  ((float)_HALF_FLOAT_SIZE_)
#define _HALF_USHORT_SIZE_MAX_ ((ushort)(_HALF_FLOAT_SIZE_))

namespace dehancer {
    
    namespace nvcc {
        
        enum class Type:int {
            i1d = 0,
            i2d,
            i3d
        };
        
        enum class PixelFormat:int {
            rgba16float = 0,
            rgba32float,
            rgba8uint,
            rgba16uint,
            rgba32uint
        };
        
        struct texture {
#ifndef CUDA_KERNEL
            __host__ [[nodiscard]] virtual const cudaArray *get_contents() const = 0;
            __host__ [[nodiscard]] virtual cudaArray *get_contents() = 0;
            __host__ [[nodiscard]] virtual const std::string& get_label() const = 0 ;
            __host__ virtual void set_label(const std::string& label) = 0;
            __host__ virtual Type get_type() = 0;
            __host__ virtual PixelFormat get_pixel_format() = 0;
#endif
            __device__ [[nodiscard]] virtual size_t get_width() const = 0;
            __device__ [[nodiscard]] virtual size_t get_height() const = 0;
            __device__ [[nodiscard]] virtual size_t get_depth() const = 0 ;
            __device__ [[nodiscard]] virtual bool is_half() const {return false;} ;
        
#ifndef CUDA_KERNEL
            __host__ virtual ~texture() = default;
#endif
        };
      
    }
}

#endif