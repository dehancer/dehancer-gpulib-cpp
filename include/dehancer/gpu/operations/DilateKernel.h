//
// Created by denn on 02.02.2021.
//

#pragma once

#include "dehancer/gpu/operations/MorphKernel.h"

namespace dehancer {
    class DilateKernel: public MorphKernel {
    public:
        using MorphKernel::MorphKernel;
    
        /***
         *
         * This operations consists of convolving an image A with some kernel ( B), which can have any shape or size,
         * usually a square or circle. The kernel B has a defined anchor point, usually being the center of the kernel.
         * As the kernel B is scanned over the image, we compute the maximal pixel value overlapped by B and replace
         * the image pixel in the anchor point position with that maximal value.
         * As you can deduce, this maximizing operation causes bright regions within an image to "grow" (therefore the name dilation).
         * The dilatation operation is: dst(x,y)=max(x′,y′):element(x′,y′)≠0src(x+x′,y+y′)
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
        explicit DilateKernel(const void *command_queue,
                             const Texture &source = nullptr,
                             const Texture &destination = nullptr,
                             size_t patches = 1,
                             size_t iterations = 1,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = ""
        );
        
        explicit DilateKernel(const void *command_queue,
                              size_t patches,
                              size_t iterations = 1,
                              bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                              const std::string &library_path = ""
        );
    };
}