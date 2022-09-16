//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/opencl/embeddedProgram.h"
#include "dehancer/gpu/Texture.h"

namespace dehancer::opencl {

    class Context {

    public:
        explicit Context(const void *command_queue);
        [[nodiscard]] cl_command_queue get_command_queue() const;
        [[nodiscard]] cl_device_id get_device_id() const;
        [[nodiscard]] cl_context get_context() const;
        [[nodiscard]] TextureInfo get_texture_info(TextureDesc::Type texture_type) const;

    private:
        const void *command_queue_;
        cl_device_id device_id_{};
        cl_context context_{};

    protected:
        mutable cl_int last_error_{};
    };
}

