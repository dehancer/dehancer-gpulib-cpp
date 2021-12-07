//
// Created by denn on 29.12.2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/texture.h"


namespace dehancer {
    
    namespace nvcc {
        
        template<class T, bool is_half_float = false>
        struct texture2d: public texture {

#ifndef CUDA_KERNEL
            __host__ [[nodiscard]] const cudaArray* get_contents() const override { return mem_; };
            __host__ [[nodiscard]] cudaArray* get_contents() override { return mem_; };
#endif
            __device__ [[nodiscard]] size_t get_width() const override { return width_;};
            __device__ [[nodiscard]] size_t get_height() const override { return height_;};
            __device__ [[nodiscard]] size_t get_depth() const override { return 1;};
            __device__ [[nodiscard]] bool is_half() const override {return is_half_; } ;

#ifndef CUDA_KERNEL
            texture2d(size_t width, size_t height, bool normalized_coords = true):
                    texture_normalized_(0),
                    surface_(0),
                    width_(width),
                    height_(height),
                    normalized_coords_(normalized_coords),
                    is_half_(is_half_float),
                    mem_(nullptr)
            {
              assert(width_ > 0 && height_ > 0);
              
              cudaChannelFormatDesc channelDesc{};
              
              if (is_half_float) {
                channelDesc = cudaCreateChannelDescHalf4();
              }
              else {
                channelDesc = cudaCreateChannelDesc<T>();
              }
              
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
              texDesc.addressMode[0]   = cudaAddressModeMirror;
              texDesc.addressMode[1]   = cudaAddressModeMirror;
              texDesc.filterMode       = cudaFilterModeLinear;
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = normalized_coords_;
              
              // Create texture object
              CHECK_CUDA(cudaCreateTextureObject(&texture_normalized_, &resDesc, &texDesc, nullptr));
            }
            
            ~texture2d() {
              if (texture_normalized_)
                cuTexObjectDestroy(texture_normalized_);
              texture_normalized_ = 0;
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
              return tex2D<T>(texture_normalized_, coords.x, coords.y);
            }

            template<class C>
            __device__
            T read(C coords) const {
              return tex2D<T>(texture_normalized_, coords.x, coords.y);
            }

            __device__
            T read_pixel(int2 gid) const {
              T data;
              int pitch = is_half_ ? sizeof(T)/2 : sizeof(T);
              surf2Dread(&data, surface_, gid.x * pitch , gid.y , cudaBoundaryModeClamp);
              return data;
            }
            
            template<class C>
            __device__
            void write(T color, C coords) {
              int pitch = is_half_ ? sizeof(T)/2 : sizeof(T);
              surf2Dwrite(color, surface_, coords.x * pitch , coords.y , cudaBoundaryModeClamp);
            }
#endif
        
        private:
            cudaTextureObject_t texture_normalized_;
            cudaSurfaceObject_t surface_;
            size_t width_;
            size_t height_;
            bool   normalized_coords_;
            bool   is_half_;

#ifndef CUDA_KERNEL
            cudaArray* mem_ = nullptr;
#endif
        };
    }
}

typedef dehancer::nvcc::texture2d<float4> image2d_t;