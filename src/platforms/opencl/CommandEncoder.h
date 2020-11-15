//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "Context.h"

namespace dehancer::opencl {

    class CommandEncoder: public dehancer::CommandEncoder {

    public:
        explicit CommandEncoder(cl_kernel kernel);
        void set(const Texture &texture, int index) override;
        void set(const void *bytes, size_t bytes_length, int index) override;

        cl_kernel kernel_ = nullptr;
    };
}
