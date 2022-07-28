//
// Created by denn nevera on 09/11/2020.
//

#pragma once

#include "dehancer/gpu/Command.h"
#include "Context.h"

namespace dehancer::opencl {

    class Command: public opencl::Context, public std::enable_shared_from_this<Command> {
    public:

        explicit Command(const void *command_queue, bool wait_until_completed = dehancer::Command::WAIT_UNTIL_COMPLETED);
        Texture make_texture(size_t width, size_t height, size_t depth);

        void enable_wait_completed(bool enable) { wait_until_completed_ = enable; };
        [[nodiscard]] bool get_wait_completed() const { return wait_until_completed_;}
        void set_wait_completed(bool value) { wait_until_completed_ = value;}
        
    private:
        bool wait_until_completed_;
    };
}
