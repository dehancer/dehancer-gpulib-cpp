//
// Created by denn nevera on 01/12/2020.
//

#pragma once

#include "dehancer/gpu/operations/UnaryKernel.h"

namespace dehancer {
    
    class OpticalReolution: public UnaryKernel {
    public:
        
        /***
         * A filter that convolves an image with a Box blur of a given channels radius in both the x and y directions.
         * @param command_queue
         * @param s - source texture
         * @param d - destination texture
         * @param radius - blur radius array by RGBA channels, array size must be 4
         * @param address_mode - the edge mode addressing to use when texture reads stray off the edge of an image
         * @param wait_until_completed - flag defines completion state
         * @param library_path - explicit shaders library file path, resource name or source bundle
         *                      (opencl source can by name of embedded value)
         */
        OpticalReolution(const void* command_queue,
                         const Texture&    s,
                         const Texture&    d,
                         std::array<float ,4> radius,
                         DHCR_EdgeAddress       address_mode = DHCR_EdgeAddress::DHCR_ADDRESS_CLAMP,
                         bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                         const std::string& library_path = ""
        );
        
        /***
         * A filter that convolves an image with a Box blur of a given radius in both the x and y directions for RGB channels only.
         */
        OpticalReolution(const void* command_queue,
                         const Texture&    s,
                         const Texture&    d,
                         float radius,
                         DHCR_EdgeAddress       address_mode = DHCR_EdgeAddress::DHCR_ADDRESS_CLAMP,
                         bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                         const std::string& library_path = ""
        );
    };
}
