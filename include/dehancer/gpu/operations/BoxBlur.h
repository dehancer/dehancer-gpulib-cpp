//
// Created by denn nevera on 01/12/2020.
//

#pragma once

#include "dehancer/gpu/operations/UnaryKernel.h"

namespace dehancer {
    
    class BoxBlur: public UnaryKernel {
    public:
        using UnaryKernel::UnaryKernel;
    
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
        BoxBlur(const void* command_queue,
                const Texture&    s,
                const Texture&    d,
                std::array<size_t ,4> radius,
                const ChannelsDesc::Transform& transform = {},
                DHCR_EdgeMode       edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                const std::string& library_path = ""
        );
        
        BoxBlur(const void* command_queue,
                std::array<size_t ,4> radius,
                const ChannelsDesc::Transform& transform = {},
                DHCR_EdgeMode       edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                const std::string& library_path = ""
        );
        
        /***
         * A filter that convolves an image with a Box blur of a given radius in both the x and y directions for RGB channels only.
         */
        BoxBlur(const void* command_queue,
                const Texture&    s,
                const Texture&    d,
                size_t radius,
                const ChannelsDesc::Transform& transform = {},
                DHCR_EdgeMode       edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                const std::string& library_path = ""
        );
        
        explicit BoxBlur(const void* command_queue,
                         size_t radius = 0,
                         const ChannelsDesc::Transform& transform = {},
                         DHCR_EdgeMode       edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                         bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                         const std::string& library_path = ""
        );
        
        [[maybe_unused]] void set_radius(size_t radius);
        [[maybe_unused]] void set_radius(std::array<size_t ,4>  radius);
        [[maybe_unused]] std::array<size_t ,4> get_radius() const;
    };
}

