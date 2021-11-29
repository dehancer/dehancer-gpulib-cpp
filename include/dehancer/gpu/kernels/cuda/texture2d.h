//
// Created by denn on 29.12.2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/texture.h"

#ifndef CUDA_KERNEL
//#include "src/platforms/cuda/Context.h"
#endif

namespace dehancer {
    
    namespace nvcc {
        
        template<class T>
        struct texture2d: public texture {

#ifndef CUDA_KERNEL
            __host__ [[nodiscard]] const cudaArray* get_contents() const override { return mem_; };
            __host__ [[nodiscard]] cudaArray* get_contents() override { return mem_; };
#endif
            __device__ [[nodiscard]] size_t get_width() const override { return width_;};
            __device__ [[nodiscard]] size_t get_height() const override { return height_;};
            __device__ [[nodiscard]] size_t get_depth() const override { return 1;};

#ifndef CUDA_KERNEL
            texture2d(size_t width, size_t height, bool normalized_coords = true):
                    texture_(0),
                    surface_(0),
                    width_(width),
                    height_(height),
                    normalized_coords_(normalized_coords)
            {
              assert(width_ > 0 && height_ > 0);
              
              cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

//            CHECK_CUDA(cudaMallocManaged(&mem_, width_*height_* sizeof(float) * 4, cudaMemAttachGlobal));
              CHECK_CUDA(
                      cudaMallocArray(&mem_, &channelDesc, width_, height_,
                                      cudaArraySurfaceLoadStore));
              
              cudaResourceDesc resDesc{};
              memset(&resDesc, 0, sizeof(resDesc));
              resDesc.resType = cudaResourceTypeArray;
              resDesc.res.array.array = mem_;
              
              //--- Specify surface ---
              CHECK_CUDA(cudaCreateSurfaceObject(&surface_, &resDesc));
              
              // Specify texture object parameters
              cudaTextureDesc texDesc{};
              memset(&texDesc, 0, sizeof(texDesc));
              texDesc.addressMode[0]   = cudaAddressModeClamp;//cudaAddressModeMirror;
              texDesc.addressMode[1]   = cudaAddressModeClamp;//cudaAddressModeMirror;
              texDesc.filterMode       = cudaFilterModeLinear;
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = normalized_coords_;
              
              // Create texture object
              CHECK_CUDA(cudaCreateTextureObject(&texture_, &resDesc, &texDesc, nullptr));
            }
            
            ~texture2d() {
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
              return tex2D<T>(texture_, coords.x, coords.y);
            }

            template<class C>
            __device__
            T read(C coords) const {
              return tex2D<T>(texture_, coords.x, coords.y);
            }

            __device__
            T read_pixel(int2 coords) const {
              T data;
              surf2Dread<T>(&data, surface_, coords.x * sizeof(T) , coords.y , cudaBoundaryModeClamp);
              return data;
            }
            
            template<class C>
            __device__
            void write(T color, C coords) {
              surf2Dwrite<T>(color, surface_, coords.x * sizeof(T) , coords.y , cudaBoundaryModeClamp);
            }
#endif
        
        private:
            cudaTextureObject_t texture_;
            cudaSurfaceObject_t surface_;
            size_t width_;
            size_t height_;
            bool   normalized_coords_;

#ifndef CUDA_KERNEL
            cudaArray* mem_ = nullptr;
#endif
        };
    }
}
typedef dehancer::nvcc::texture2d<float4> image2d_t;