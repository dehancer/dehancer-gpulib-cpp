//
// Created by denn nevera on 2019-08-02.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/kernels/types.h"

namespace dehancer {
    
    /**
     * Resample kernel
     */
    class ResampleKernel: public Kernel {
    
    public:
        using Kernel::Kernel;
    
        /***
         * Resample mode
         */
        enum Mode {
            bilinear = DHCR_Bilinear ,
            bicubic  = DHCR_Bicubic,
            box_average = DHCR_BoxAverage
        };
        
        explicit ResampleKernel(const void *command_queue,
                                const Texture &source,
                                const Texture &destination,
                                Mode mode = bilinear,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");
        
        explicit ResampleKernel(const void *command_queue,
                                Mode mode = bilinear,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");
    
        [[maybe_unused]] void set_mode(float Mode);
    };
}