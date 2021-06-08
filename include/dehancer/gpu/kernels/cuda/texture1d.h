//
// Created by denn on 29.12.2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/texture.h"
#ifndef CUDA_KERNEL
#include "dehancer/gpu/Memory.h"
#endif

namespace dehancer {
    
    namespace nvcc {
        
        template<class T>
        struct texture1d: public texture {

#ifndef CUDA_KERNEL
            __host__ [[nodiscard]] const cudaArray* get_contents() const override { return mem_; };
            __host__ [[nodiscard]] cudaArray* get_contents() override { return mem_; };
#endif
            __device__ [[nodiscard]] size_t get_width() const override { return width_;};
            __device__ [[nodiscard]] size_t get_height() const override { return 1;}
            __device__ [[nodiscard]] size_t get_depth() const override { return 1;}

#ifndef CUDA_KERNEL
            explicit texture1d(size_t width, const Memory& mem = nullptr, bool normalized_coords = true):
                    texture(),
                    texture_(0),
                    surface_(0),
                    width_(width),
                    normalized_coords_(normalized_coords),
                    mem_(nullptr)
            {
              assert(width_ > 0);
              
              cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
              cudaResourceDesc resDesc{};
              memset(&resDesc, 0, sizeof(resDesc));
              
              if (!mem) {
                CHECK_CUDA(
                        cudaMallocArray(&mem_, &channelDesc, width_, 1,
                                        cudaArraySurfaceLoadStore));
                
                resDesc.resType = cudaResourceTypeArray;
                resDesc.res.array.array = mem_;
              }
              else {
                resDesc.resType = cudaResourceTypeLinear;
                resDesc.res.linear.devPtr = mem->get_memory();
                resDesc.res.linear.desc = channelDesc;
                resDesc.res.linear.sizeInBytes = mem->get_length();
              }
              
              //--- Specify surface ---
              CHECK_CUDA(cudaCreateSurfaceObject(&surface_, &resDesc));
              
              // Specify texture object parameters
              cudaTextureDesc texDesc{};
              memset(&texDesc, 0, sizeof(texDesc));
              texDesc.addressMode[0]   = cudaAddressModeMirror;
              texDesc.addressMode[1]   = cudaAddressModeMirror;
              texDesc.filterMode       = cudaFilterModeLinear;
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = normalized_coords_;
              
              // Create texture object
              CHECK_CUDA(cudaCreateTextureObject(&texture_, &resDesc, &texDesc, nullptr));
            }
            
            ~texture1d() {
              if (texture_)
                cuTexObjectDestroy(texture_);
              texture_ = 0;
              if (surface_)
                cuSurfObjectDestroy(surface_);
              surface_ = 0;
              if (mem_)
                cudaFreeArray(mem_);
              mem_ = nullptr;
            }

#else
            template<class C>
            __device__
            T read(C coord) {
              return tex1D<T>(texture_, coord);
            }

              template<class C>
            __device__
            T read(C coord) const {
              return tex1D<T>(texture_, coord);
            }

//
//
//             1D DOES NOT WORK! I don't know why...
//
//             __device__
//            T read_pixel(int coords) const {
//              T data;
//              surf1Dread<T>(&data, surface_, coords* sizeof(T) , cudaBoundaryModeClamp);
//              return data;
//            }
//
//            template<class C>
//            __device__
//            void write(T color, C coord) {
//              surf1Dwrite<T>(color, surface_, coord * sizeof(T) , cudaBoundaryModeClamp);
//            }

            __device__
            T read_pixel(int coord) const {
              T data;
              surf2Dread<T>(&data, surface_, coord * sizeof(T) , 0 , cudaBoundaryModeClamp);
              return data;
            }
            
            template<class C>
            __device__
            void write(T color, C coord) {
              surf2Dwrite<T>(color, surface_, coord * sizeof(T) , 0 , cudaBoundaryModeClamp);
            }
#endif
        
        private:
            cudaTextureObject_t texture_;
            cudaSurfaceObject_t surface_;
            size_t width_;
            bool normalized_coords_;

#ifndef CUDA_KERNEL
            cudaArray_t mem_ = nullptr;
#endif
        };
    }
}

typedef dehancer::nvcc::texture1d<float4> image1d_t;
