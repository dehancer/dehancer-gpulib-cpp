//
// Created by denn on 29.12.2020.
//

#pragma once

#include "dehancer/gpu/kernels/cuda/texture.h"

namespace dehancer {

    namespace nvcc {

        template<class T, bool is_half_float = false>
        struct texture3d: public texture {

#ifndef CUDA_KERNEL
            __host__ [[nodiscard]] const cudaArray* get_contents() const override { return mem_; };
            __host__ [[nodiscard]] cudaArray* get_contents() override { return mem_; };
            __host__ [[nodiscard]] const std::string& get_label() const override {return label_;};
            __host__ void set_label(const std::string& label) override {label_ = label;};
#endif
            __device__ [[nodiscard]] size_t get_width() const override { return width_;};
            __device__ [[nodiscard]] size_t get_height() const override { return height_;};
            __device__ [[nodiscard]] size_t get_depth() const override { return depth_;};

#ifndef CUDA_KERNEL
            texture3d(size_t width, size_t height, size_t depth, bool normalized_coords = true):
                    texture_(0),
                    surface_(0),
                    width_(width),
                    height_(height),
                    depth_(depth),
                    normalized_coords_(normalized_coords),
                    is_half_(is_half_float),
                    pitch_(sizeof(T)),
                    mem_(nullptr)
            {
              assert(width_ > 0 && height_ > 0 && depth_ > 0);

              cudaChannelFormatDesc channelDesc{}; //= cudaCreateChannelDesc<T>();
  
              if (is_half_) {
                channelDesc = cudaCreateChannelDesc<ushort4>();
              }
              else {
                channelDesc = cudaCreateChannelDesc<T>();
              }

              CHECK_CUDA(cudaMalloc3DArray(&mem_, &channelDesc, {width_,height_,depth_}, cudaArraySurfaceLoadStore));

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
              texDesc.addressMode[2]   = cudaAddressModeMirror;
              /***
               * ALWAYS LINEAR! IT USES for LUT interpolations only
               */
              //texDesc.filterMode       = cudaFilterModeLinear;
  
              if (is_half_)
                texDesc.filterMode       = cudaFilterModePoint;
              else
                texDesc.filterMode       = cudaFilterModeLinear;
  
              texDesc.readMode         = cudaReadModeElementType;
              texDesc.normalizedCoords = normalized_coords_;

              // Create texture object
              CHECK_CUDA(cudaCreateTextureObject(&texture_, &resDesc, &texDesc, nullptr));
  
              //rintf("texture3d %zux%zux%zu is_half_ = %i\n", width_, height_, depth_, is_half_);
  
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
            T read_text_ushort4(C coords) const {
                auto d = tex3D<ushort4>(texture_, coords.x, coords.y, coords.z);
                return float4({
                  clamp((float)(d.x)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(d.y)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(d.z)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(d.w)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f)
                });
            }
            
//            template<class C>
//            __device__
//            T read(C coords) {
//              return tex3D<T>(texture_, coords.x, coords.y, coords.z);
//            }

            template<class C>
            __device__
            T read(C coords) {
              return is_half_
              ?
              read_text_ushort4(coords)
              :
              tex3D<T>(texture_, coords.x, coords.y, coords.z);
              
            }
            
//            template<class C>
//            __device__
//            T read(C coords) const {
//              return tex3D<T>(texture_, coords.x, coords.y, coords.z);
//            }

            template<class C>
            __device__
            T read(C coords) const {
              return is_half_
              ?
              read_text_ushort4(coords)
              :
              tex3D<T>(texture_, coords.x, coords.y, coords.z);
              
            }
            
//            __device__
//            T read_pixel(int3 coords) const {
//              T data;
//              surf3Dread<T>(&data, surface_, coords.x * sizeof(T) , coords.y ,  coords.z , cudaBoundaryModeClamp);
//              return data;
//            }
            
            __device__
            T read_pixel(int3 gid) const {
              T data;
              if (is_half_) {
                ushort4 uc;
                surf3Dread(&uc, surface_, gid.x * sizeof(ushort4) , gid.y,  gid.z , cudaBoundaryModeClamp);
                data = float4({
                  clamp((float)(uc.x)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(uc.y)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(uc.z)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f),
                  clamp((float)(uc.w)/_HALF_FLOAT_SIZE_MAX_, 0.0f, 1.0f)
                });
              }
              else {
                surf3Dread(&data, surface_, gid.x * pitch_, gid.y,  gid.z, cudaBoundaryModeClamp);
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
                surf3Dwrite(uc, surface_, coords.x * sizeof(ushort4) , coords.y ,  coords.z ,cudaBoundaryModeClamp);
            }
            
//            template<class C>
//            __device__
//            void write(T color, C coords) {
//              surf3Dwrite<T>(color, surface_, coords.x * sizeof(T) , coords.y,  coords.z , cudaBoundaryModeClamp);
//            }

            template<class C>
            __device__
            void write(T color, C coords) {
              is_half_
              ?
              write_ushort4(color, coords)
              :
              surf3Dwrite(color, surface_, coords.x * pitch_ , coords.y,  coords.z , cudaBoundaryModeClamp);
            }
#endif

        private:
            cudaTextureObject_t texture_{};
            cudaSurfaceObject_t surface_{};
            size_t width_;
            size_t height_;
            size_t depth_;
            bool normalized_coords_;
            bool   is_half_;
            size_t pitch_;

#ifndef CUDA_KERNEL
            cudaArray* mem_ = nullptr;
            std::string label_{};
#endif
        };
    }
}

typedef dehancer::nvcc::texture3d<float4> image3d_t;