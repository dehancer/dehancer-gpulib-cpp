//
// Created by denn on 29.12.2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/utils.h"
#include "dehancer/gpu/kernels/cuda/texture.h"

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
            explicit texture1d(size_t width):
                    texture(),
                    texture_(0),
                    surface_(0),
                    width_(width)
            {
              assert(width_ > 0);

              cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

              CHECK_CUDA(
                      cudaMallocArray(&mem_, &channelDesc, width_, 1,
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
              texDesc.addressMode[0]   = cudaAddressModeClamp;
              texDesc.addressMode[1]   = cudaAddressModeMirror;
              texDesc.filterMode       = cudaFilterModeLinear;
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = 1;

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
              return tex2D<T>(texture_, coord, 0);
            }

            template<class C>
            __device__
            void write(T color, C coord) {
              surf2Dwrite<T>(color, surface_, coord * sizeof(T) , 0, cudaBoundaryModeClamp);
            }
#endif

        private:
            cudaTextureObject_t texture_;
            cudaSurfaceObject_t surface_;
            size_t width_;

#ifndef CUDA_KERNEL
            cudaArray* mem_ = nullptr;
#endif
        };
    }
}
