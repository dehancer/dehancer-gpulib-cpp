//
// Created by denn on 29.12.2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/texture.h"
#include <cuda_runtime_api.h>

#define _HALF_FLOAT_SIZE_BASE_ (65535)
#define _HALF_FLOAT_SIZE_      (_HALF_FLOAT_SIZE_BASE_>>1)
#define _HALF_FLOAT_SIZE_MAX_  ((float)_HALF_FLOAT_SIZE_)
#define _HALF_USHORT_SIZE_MAX_ ((ushort)(_HALF_FLOAT_SIZE_))

namespace dehancer {
    
    namespace nvcc {
        
        template<class T, bool is_half_float = false>
        struct texture2d: public texture {

#ifndef CUDA_KERNEL
            
            __host__ [[nodiscard]] const cudaArray* get_contents() const override { return mem_; };
            __host__ [[nodiscard]] cudaArray* get_contents() override { return mem_; };
            __host__ [[nodiscard]] const std::string& get_label() const override {return label_;};
            __host__ void set_label(const std::string& label) override {label_ = label;};
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
                    mem_(nullptr),
                    pitch_(sizeof(T))
            {
              assert(width_ > 0 && height_ > 0);
              
              cudaChannelFormatDesc channelDesc{};
              
              if (is_half_) {
                channelDesc = cudaCreateChannelDesc<ushort4>();
              }
              else {
                channelDesc = cudaCreateChannelDesc<T>();
              }
              
              try {
                CHECK_CUDA(
                        cudaMallocArray(&mem_,
                                        &channelDesc,
                                        width_,
                                        height_,
                                        cudaArraySurfaceLoadStore));
              }
              
              catch (std::runtime_error &e) {
                throw std::runtime_error(dehancer::error_string("texture: %ix%i type: %i %s\n", width_, height_, is_half_, e.what()));
              }
              
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
              
              if (is_half_)
                texDesc.filterMode       = cudaFilterModePoint;
              else
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
            T read_text_ushort4(C coords) const {
                auto d = tex2D<ushort4>(texture_normalized_, coords.x, coords.y);
                return float4({
                  clamp((float)(d.x)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(d.y)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(d.z)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(d.w)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f)
                });
            }
      
            template<class C>
            __device__
            T read(C coords) {
              return is_half_
              ?
              read_text_ushort4(coords)
              :
              tex2D<T>(texture_normalized_, coords.x, coords.y);
              
            }

             template<class C>
            __device__
            T read(C coords) const {
              return is_half_
                ?
                read_text_ushort4(coords)
                :
                tex2D<T>(texture_normalized_, coords.x, coords.y);
            }
            
            __device__
            T read_pixel(int2 gid) const {
              T data;
              if (is_half_) {
                ushort4 uc;
                surf2Dread(&uc, surface_, gid.x * sizeof(ushort4) , gid.y , cudaBoundaryModeClamp);
                data = float4({
                  clamp((float)(uc.x)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(uc.y)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(uc.z)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(uc.w)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f)
                });
              }
              else {
                surf2Dread(&data, surface_, gid.x * pitch_, gid.y, cudaBoundaryModeClamp);
              }
              return data;
            }
            
             template<class C>
            __device__
            void write_ushort4(T color, C coords) {
                ushort4 uc =  ushort4({
                  clamp((ushort)(color.x*_HALF_FLOAT_SIZE_MAX_), (ushort)0, (ushort)_HALF_FLOAT_SIZE_BASE_),
                  clamp((ushort)(color.y*_HALF_FLOAT_SIZE_MAX_), (ushort)0, (ushort)_HALF_FLOAT_SIZE_BASE_),
                  clamp((ushort)(color.z*_HALF_FLOAT_SIZE_MAX_), (ushort)0, (ushort)_HALF_FLOAT_SIZE_BASE_),
                  clamp((ushort)(color.w*_HALF_FLOAT_SIZE_MAX_), (ushort)0, (ushort)_HALF_FLOAT_SIZE_BASE_)
                });
                surf2Dwrite(uc, surface_, coords.x * sizeof(ushort4) , coords.y , cudaBoundaryModeClamp);
            }
            
            template<class C>
            __device__
            void write(T color, C coords) {
              is_half_
              ?
              write_ushort4(color, coords)
              :
              surf2Dwrite(color, surface_, coords.x * pitch_ , coords.y , cudaBoundaryModeClamp);
            }
#endif
        
        private:
            cudaTextureObject_t texture_normalized_;
            cudaSurfaceObject_t surface_;
            size_t width_;
            size_t height_;
            bool   normalized_coords_;
            bool   is_half_;
            size_t pitch_;

#ifndef CUDA_KERNEL
            cudaArray* mem_ = nullptr;
            std::string label_;
#endif
        };
    }
}

typedef dehancer::nvcc::texture2d<float4> image2d_t;