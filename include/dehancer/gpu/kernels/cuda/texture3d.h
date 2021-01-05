//
// Created by denn on 29.12.2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/texture.h"

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
            texture3d(size_t width, size_t height, size_t depth):
                    texture_(0),
                    surface_(0),
                    width_(width),
                    depth_(depth),
                    height_(height)
            {
              assert(width_ > 0 && height_ > 0 && depth_ > 0);

              cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

              CHECK_CUDA(cudaMalloc3DArray(&mem_, &channelDesc, {width_,height_,depth_}));

              cudaResourceDesc resDesc{};
              memset(&resDesc, 0, sizeof(resDesc));
              resDesc.resType = cudaResourceTypeArray;
              resDesc.res.array.array = mem_;

              //--- Specify surface ---
              CHECK_CUDA(cudaCreateSurfaceObject(&surface_, &resDesc));

              // Specify texture object parameters
              cudaTextureDesc texDesc{};
              memset(&texDesc, 0, sizeof(texDesc));
              texDesc.addressMode[0]   = cudaAddressModeMirror;//cudaAddressModeClamp;
              texDesc.addressMode[1]   = cudaAddressModeMirror;//cudaAddressModeClamp;
              texDesc.addressMode[2]   = cudaAddressModeMirror;//cudaAddressModeClamp;
              texDesc.filterMode       = cudaFilterModeLinear;
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = 1;

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

#ifndef CUDA_KERNEL
            cudaArray* mem_ = nullptr;
#endif
        };
    }
}

typedef dehancer::nvcc::texture3d<float4> image3d_t;