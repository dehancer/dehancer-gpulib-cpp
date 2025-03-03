//
// Created by denn on 09.01.2021.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/operations/PassKernel.h"
#include "dehancer/gpu/Channels.h"
#include <vector>
#include <any>

namespace dehancer {
    
    struct UnaryKernelImpl;
    
    /***
     * Base Kernel operation class
     */
    class UnaryKernel: public PassKernel {
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
        
        /***
         * Kernel function must create kernel line for channel with a chosen index.
         * For example:
         *   struct BoxBlurOptions {
         *        std::array<size_t, 4> radius_array;
         *    };
         *
         *
         *  auto kernel_box_blur = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
         *
         *       data.clear();
         *
         *       if (!user_data.has_value()) return ;
         *
         *       auto options = std::any_cast<BoxBlurOptions>(user_data.value());
         *
         *       auto radius = options.radius_array.at(index);
         *
         *       if (radius <= 1 ) return;
         *       for (int i = 0; i < radius; ++i) {
         *         data.push_back(1.0f/(float)radius);
         *       }
         *
         *       return 1.0f; // return scale factor to increase convolution perform
         *   };
         */
        using KernelFunction = std::function<float (int channel_index, std::vector<float>& line, const UserData& user_data)>;
        
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
            
            float          amplify = 1.0f;
            
            /***
             * The edge mode to use when texture reads stray off the edge of an image.
             * Most kernel objects can read off the edge of a source image.
             * This can happen because of a negative offset property, because the offset + clipRect.size is larger than
             * the source image, or because the filter uses neighboring pixels in its calculations (e.g. convolution filters).
             */
            DHCR_EdgeMode    edge_mode = DHCR_EdgeMode::DHCR_ADDRESS_CLAMP;
    
            /***
             * Mask unary operation
             */
//            Texture           mask = nullptr;
        };
        
        using PassKernel::PassKernel;
        
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
                    const ChannelsDesc::Transform& transform = {},
                    bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                    const std::string& library_path = ""
        );
        
        UnaryKernel(const void* command_queue,
                    const Options& options,
                    const ChannelsDesc::Transform& transform = {},
                    bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                    const std::string& library_path = ""
        );
        
        /***
         * Process current source
         */
        void process() override;
        
        /***
         * Process a new source to a new destination
         * @param source
         * @param destination
         */
        void process(const Texture& source, const Texture& destination) override;
    
        /***
         * Set new source
         * @param source
         */
        [[maybe_unused]] void set_source(const Texture& source) override;
        
        /***
         * Set new destination
         * @param destination
         */
        [[maybe_unused]] void set_destination(const Texture& destination) override;
        
        /***
         * Set edge mode
         * @param mode
         */
        [[maybe_unused]] void set_edge_mode(DHCR_EdgeMode mode);
        
        /***
         * Set channel colors transformation
         * @param transform
         */
        [[maybe_unused]] void set_transform(const ChannelsDesc::Transform &transform);
        
        /***
         * Set unary mask
         * @param mask
         */
        //[[maybe_unused]] void set_mask(const Texture &mask);
    
        /***
        * Set unary amplify
        * @param amplify
        */
        [[maybe_unused]] void set_amplify(float amplify);
    
        
        /***
         * Get current channel colors transformation
         * @return
         */
        const ChannelsDesc::Transform & get_transform() const;
        
    protected:
        
        /***
         * Set new unary kernel options
         * @param options
         */
        [[maybe_unused]] virtual void set_options(const Options& options);
        
        /***
         * Get current options
         * @return
         */
        virtual const Options& get_options() const;
        virtual Options& get_options();
        
        /***
         * Set user data
         * @param user_data
         */
        virtual void set_user_data(const UserData &user_data);
    
    private:
        std::shared_ptr<UnaryKernelImpl> impl_;
    };
}
