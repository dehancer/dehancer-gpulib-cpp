//
// Created by denn on 02.02.2021.
//

#pragma once

#include "dehancer/gpu/operations/MorphKernel.h"

namespace dehancer {
    
    class ErodeKernel: public MorphKernel {
    public:
        using MorphKernel::MorphKernel;
    
        /***
         *
         * This operation is the sister of dilation.
         * It computes a local minimum over the area of given kernel.
         * As the kernel B is scanned over the image, we compute the minimal pixel value overlapped by B
         * and replace the image pixel under the anchor point with that minimal value.
         * The erosion operation is: dst(x,y)=min(x′,y′):element(x′,y′)≠0src(x+x′,y+y′)
         *
         * @param command_queue - platform based command queue
         * @param source - source kernel texture
         * @param destination - destination texture
         * @param patches - how many patches must be dilated around point, i.e. ---*---, where - is one patch
         * @param iterations - how many times apply dilation
         * @param wait_until_completed - flag defines completion state
         * @param library_path - explicit shaders library file path, resource name or source bundle
         *                      (opencl source can by name of embedded value)
         */
        explicit ErodeKernel(const void *command_queue,
                             const Texture &source = nullptr,
                             const Texture &destination = nullptr,
                             size_t patches = 1,
                             size_t iterations = 1,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = ""
        );
        
        explicit ErodeKernel(const void *command_queue,
                              size_t patches,
                              size_t iterations = 1,
                              bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                              const std::string &library_path = ""
        );
    };
}