//
// Created by denn on 29.12.2020.
//

#ifndef DEHANCER_GPULIB_TEXTURE_H
#define DEHANCER_GPULIB_TEXTURE_H

namespace dehancer {
    namespace nvcc {

        template<class T>
        struct texture2d {
            cudaTextureObject_t texture;
            cudaSurfaceObject_t surface;
            size_t width;
            size_t height;

            __device__ [[nodiscard]] size_t get_width() const { return width;};
            __device__ [[nodiscard]] size_t get_height() const { return height;};

#ifdef CUDA_KERNEL
            __device__
            void write(T color, uint2 coords) {
              surf2Dwrite<T>(color, surface, coords.x * sizeof(T) , coords.y , cudaBoundaryModeClamp);
            };
#endif
        };
    }
}

#endif //DEHANCER_GPULIB_TEXTURE_H
