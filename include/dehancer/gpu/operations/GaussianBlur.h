//
// Created by denn nevera on 01/12/2020.
//

#pragma once

#include "dehancer/gpu/operations/UnaryKernel.h"

namespace dehancer {
    
    class GaussianBlur: public UnaryKernel {
    public:
    
        using UnaryKernel::UnaryKernel;
    
        static constexpr float accuracy = 0.001;
        
        /***
         * A filter that convolves an image with a Gaussian blur of a given channels radius in both the x and y directions.
         * @param command_queue
         * @param s - source texture
         * @param d - destination texture
         * @param radius - blur radius array by RGBA channels, array size must be 4
         * @param address_mode - the edge mode addressing to use when texture reads stray off the edge of an image
         * @param accuracy - performance accuracy
         * @param wait_until_completed - flag defines completion state
         * @param library_path - explicit shaders library file path, resource name or source bundle
         *                      (opencl source can by name of embedded value)
         */
        GaussianBlur(const void* command_queue,
                     const Texture&    s,
                     const Texture&    d,
                     std::array<float,4> radius,
                     const ChannelsDesc::Transform& transform = {},
                     DHCR_EdgeMode   edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                     float            accuracy = GaussianBlur::accuracy,
                     bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                     const std::string& library_path = ""
        );
        
        GaussianBlur(const void* command_queue,
                     std::array<float,4> radius,
                     const ChannelsDesc::Transform& transform = {},
                     DHCR_EdgeMode   edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                     float            accuracy = GaussianBlur::accuracy,
                     bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                     const std::string& library_path = ""
        );
        
        /***
         * A filter that convolves an image with a Gaussian blur of a given radius in both the x and y directions for RGB channels only.
         */
        GaussianBlur(const void* command_queue,
                     const Texture&    s,
                     const Texture&    d,
                     float radius,
                     const ChannelsDesc::Transform& transform = {},
                     DHCR_EdgeMode   edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                     float            accuracy = 0.001,
                     bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                     const std::string& library_path = ""
        );
        
        explicit GaussianBlur(const void* command_queue,
                              float radius = 0,
                              const ChannelsDesc::Transform& transform = {},
                              DHCR_EdgeMode   edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP,
                              float            accuracy = 0.001,
                              bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                              const std::string& library_path = ""
        );
        
        [[maybe_unused]] void set_radius(float radius);
        [[maybe_unused]] void set_radius(std::array<float ,4>  radius);
        [[maybe_unused]] void set_accuracy(float accuracy);
    
        void set_options(const Options &options) override;
        
        [[maybe_unused]] std::array<float ,4> get_radius() const;
    };
}

