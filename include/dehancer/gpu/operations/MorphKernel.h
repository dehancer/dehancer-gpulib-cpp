//
// Created by denn on 02.02.2021.
//

#pragma once

#include "dehancer/gpu/Kernel.h"

namespace dehancer {
    
    class MorphKernel: public Kernel {
    public:
        
        /***
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
        explicit MorphKernel(const void *command_queue,
                             const std::string& morph_kernel_name,
                             const Texture &source = nullptr,
                             const Texture &destination = nullptr,
                             size_t patches = 1,
                             size_t iterations = 1,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = ""
        );
        
        void process() override;
        void process(const Texture &source, const Texture &destination) override;
        
        void set_size(size_t patches);
        void set_iterations(size_t iterations);
        
        void set_destination(const Texture &destination) override;
        
    private:
        size_t patches_;
        size_t iterations_;
        Texture tmp_;
    };
}
