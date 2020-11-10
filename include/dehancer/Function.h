//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/Command.h"

namespace dehancer {

    namespace impl {
        class Function;
    }

    class CommandEncoder {
    public:
        virtual void set(const Texture& texture, int index) = 0;
        virtual void set(const void* bytes, size_t bytes_length, int index) = 0;
    };

    class Function: public Command {
    public:
        typedef std::function<Texture (CommandEncoder& compute_encoder)> FunctionHandler;
        Function(const void *command_queue, const std::string& kernel_name, bool wait_until_completed = WAIT_UNTIL_COMPLETED);
        void execute(const FunctionHandler& block);

    protected:
        std::shared_ptr<impl::Function> impl_;
    };
}


