//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#import <Metal/Metal.h>

namespace dehancer::metal {

    class Context {

    public:
        explicit Context(const void *command_queue);
        [[nodiscard]] id<MTLCommandQueue> get_command_queue() const;
        [[nodiscard]] id<MTLDevice> get_device() const;

    private:
        const void* command_queue_;
    };
}
