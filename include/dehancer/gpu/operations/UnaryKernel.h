//
// Created by denn on 09.01.2021.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/Channels.h"
#include <vector>
#include <any>

namespace dehancer {
    
    struct UnaryKernelImpl;
    
    /***
     * Base Kernel operation class
     */
    class UnaryKernel: public ChannelsInput {
    public:
        
        /**
         * Separable Row/Column function must be defined.
         * For example:
         *  box = 1/9 * [1 1 1 ...]' x [1 1 1 ...] is: 9x9 kernel weights matrix
         *                             1/9  ... 1/9
         *                             1/9  ... 1/9
         *                             ...      ...
         *                             1/9 ...  1/9
         */
        
        using UserData = std::optional<std::any>;
        
        using KernelFunction = std::function<void (int channel_index, std::vector<float>& line, const UserData& user_data)>;
        
        /***
         * A structure defines options to process convolve with UnaryKernel class
         */
        struct Options {
            /***
             * A function handler computes row kernel line
             */
            KernelFunction row;
            /***
             * A function handler computes column kernel line
             */
            KernelFunction col;
            /***
             * User defined data can be used by row and column function handlers
             */
            UserData       user_data = std::nullopt;
            /***
             * The edge mode to use when texture reads stray off the edge of an image.
             * Most kernel objects can read off the edge of a source image.
             * This can happen because of a negative offset property, because the offset + clipRect.size is larger than
             * the source image, or because the filter uses neighboring pixels in its calculations (e.g. convolution filters).
             */
            DHCR_EdgeAddress    address_mode = DHCR_EdgeAddress::DHCR_ADDRESS_CLAMP;
        };
    
        /***
         * A filter that convolves an image with a given kernel of odd width and height that must be defined
         * in the options constructor parameter.
         *
         * Filter width and height can be either 3, 5, 7 or 9.
         * If there are multiple channels in the source image, each channel is processed independently.
         * A separable convolution filter may perform better when done in two passes.
         * A convolution filter is separable if the ratio of filter values between all rows is constant over the whole row.
         *
         * @param command_queue - platform based command queue
         * @param s - source texture
         * @param d - destination texture
         * @param options - convolve options
         * @param wait_until_completed - flag defines completion state
         * @param library_path - explicit shaders library file path, resource name or source bundle
         *                      (opencl source can by name of embedded value)
         */
        UnaryKernel(const void* command_queue,
                    const Texture& s,
                    const Texture& d,
                    const Options& options,
                    bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                    const std::string& library_path = ""
        );

        void process() override;
        
    private:
        std::shared_ptr<UnaryKernelImpl> impl_;
    };
}