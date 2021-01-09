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
        
        struct Options {
            KernelFunction row;
            KernelFunction col;
            UserData       user_data = std::nullopt;
            EdgeAddress    address_mode = EdgeAddress::ADDRESS_CLAMP;
        };
    
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
