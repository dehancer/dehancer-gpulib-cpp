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
        struct texture3d: public texture {

#ifndef CUDA_KERNEL
            __host__ [[nodiscard]] const cudaArray* get_contents() const override { return mem_; };
            __host__ [[nodiscard]] cudaArray* get_contents() override { return mem_; };
#endif
            __device__ [[nodiscard]] size_t get_width() const override { return width_;};
            __device__ [[nodiscard]] size_t get_height() const override { return height_;};
            __device__ [[nodiscard]] size_t get_depth() const override { return depth_;};

#ifndef CUDA_KERNEL
            texture3d(size_t width, size_t height, size_t depth, const Memory& mem = nullptr, bool normalized_coords = true):
                    texture_(0),
                    surface_(0),
                    width_(width),
                    depth_(depth),
                    height_(height),
                    normalized_coords_(normalized_coords),
                    mem_(nullptr)
            {
              assert(width_ > 0 && height_ > 0 && depth_ > 0);
              
              cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
              
              cudaResourceDesc resDesc{};
              memset(&resDesc, 0, sizeof(resDesc));
              
              if (!mem) {
                CHECK_CUDA(
                        cudaMalloc3DArray(&mem_, &channelDesc, {width_, height_, depth_}, cudaArraySurfaceLoadStore));
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
              texDesc.addressMode[2]   = cudaAddressModeMirror;
              /***
               * ALWAYS LINEAR! IT USES for LUT interpolations only
               */
              texDesc.filterMode       = cudaFilterModeLinear;
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = normalized_coords_;
              
              // Create texture object
              CHECK_CUDA(cudaCreateTextureObject(&texture_, &resDesc, &texDesc, nullptr));
            }
            
            ~texture3d() {
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
            T read(C coords) {
              return tex3D<T>(texture_, coords.x, coords.y, coords.z);
            }

            template<class C>
            __device__
            T read(C coords) const {
              return tex3D<T>(texture_, coords.x, coords.y, coords.z);
            }

            __device__
            T read_pixel(int3 coords) const {
              T data;
              surf3Dread<T>(&data, surface_, coords.x * sizeof(T) , coords.y ,  coords.z , cudaBoundaryModeClamp);
              return data;
            }
            
            template<class C>
            __device__
            void write(T color, C coords) {
              surf3Dwrite<T>(color, surface_, coords.x * sizeof(T) , coords.y,  coords.z , cudaBoundaryModeClamp);
            }
#endif
        
        private:
            cudaTextureObject_t texture_;
            cudaSurfaceObject_t surface_;
            size_t width_;
            size_t height_;
            size_t depth_;
            bool normalized_coords_;

#ifndef CUDA_KERNEL
            cudaArray_t mem_ = nullptr;
#endif
        };
    }
}

typedef dehancer::nvcc::texture3d<float4> image3d_t;