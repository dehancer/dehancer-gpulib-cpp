//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/Typedefs.h"
#include "Context.h"

namespace dehancer::metal {

    class CommandEncoder: public dehancer::CommandEncoder {

    public:
        explicit CommandEncoder(id<MTLComputeCommandEncoder> command_encoder): command_encoder_(command_encoder){}

        void set(const Texture &texture, int index) override;
        void set(const Memory& memory, int index) override;
        void set(const void *bytes, size_t bytes_length, int index) override;

        id<MTLComputeCommandEncoder> command_encoder_ = nullptr;
    };
}
