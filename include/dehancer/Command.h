//
// Created by denn nevera on 04/10/2020.
//

#pragma once

//#include "GpuConfig.h"

#include <string>
#include <memory>
#include <functional>

#include "dehancer/Texture.h"

namespace dehancer {

    namespace impl { class Command; }

    class Command {

    public:
        static bool WAIT_UNTIL_COMPLETED;

        explicit Command(const void *command_queue, bool wait_until_completed = WAIT_UNTIL_COMPLETED);
        Texture make_texture(size_t width, size_t height, size_t depth = 1);

        virtual void enable_wait_completed(bool enable);
        virtual bool get_wait_completed();

        [[nodiscard]] const void* get_command_queue() const;
        void* get_command_queue();

    protected:
        std::shared_ptr<impl::Command> impl_;
    };

}

