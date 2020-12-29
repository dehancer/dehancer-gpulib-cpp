//
// Created by denn on 29.12.2020.
//

#pragma once

#include <dehancer/gpu/kernels/cuda/utils.h>

namespace dehancer {

    namespace nvcc {

        template<class T>
        struct texture2d {

            __host__ [[nodiscard]] cudaArray* get_contents() const { return mem_; };
            __device__ [[nodiscard]] size_t get_width() const { return width_;};
            __device__ [[nodiscard]] size_t get_height() const { return height_;};

#ifndef CUDA_KERNEL
            texture2d(size_t width,size_t height):
                    width_(width),
                    height_(height)
            {
              assert(width_ > 0 && height_ > 0);

              cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

              CHECK_CUDA(
                      cudaMallocArray(&mem_, &channelDesc, width_, height_,
                                      cudaArraySurfaceLoadStore));

              cudaResourceDesc resDesc{};
              memset(&resDesc, 0, sizeof(resDesc));
              resDesc.resType = cudaResourceTypeArray;
              resDesc.res.array.array = mem_;

              //--- Specify surface ---
              CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resDesc));

              // Specify texture object parameters
              cudaTextureDesc texDesc{};
              memset(&texDesc, 0, sizeof(texDesc));
              texDesc.addressMode[0]   = cudaAddressModeClamp;
              texDesc.addressMode[1]   = cudaAddressModeClamp;
              texDesc.filterMode       = cudaFilterModeLinear;
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = 1;

              // Create texture object
              CHECK_CUDA(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
            }

#else
            template<class C>
            __device__
            T read(C coords) {
              return tex2D<float4>(texture, coords.x, coords.y);
            };

            template<class C>
            __device__
            void write(T color, C coords) {
              surf2Dwrite<T>(color, surface, coords.x * sizeof(T) , coords.y , cudaBoundaryModeClamp);
            };
#endif

        private:
            cudaTextureObject_t texture{};
            cudaSurfaceObject_t surface{};
            size_t width_{};
            size_t height_{};

#ifndef CUDA_KERNEL
            cudaArray* mem_{};
#endif
        };
    }
}
