//
// Created by denn nevera on 10/11/2020.
//

#pragma once

//@protocol MTLCommandQueue;
//@protocol MTLDevice;

namespace dehancer::metal {
    
    class Context {

    public:
        explicit Context(const void *command_queue);
        //[[nodiscard]] id<MTLCommandQueue> get_command_queue() const;
        //[[nodiscard]] id<MTLDevice> get_device() const;
        [[nodiscard]] void* get_command_queue() const;
        [[nodiscard]] void* get_device() const;
        [[nodiscard]] bool has_unified_memory() const;

    private:
        const void* command_queue_;
    };
}

