//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/Typedefs.h"
#include "Context.h"
#include "Command.h"

namespace dehancer::metal {

    class CommandEncoder: public dehancer::CommandEncoder {

    public:
        using dehancer::CommandEncoder::CommandEncoder;
        
        explicit CommandEncoder(dehancer::metal::Command* command_queue, void* pipeline, void* command_encoder):
        command_queue_(command_queue),
        pipeline_(pipeline),
        command_encoder_(command_encoder){}

        void set(const Texture &texture, int index) override;
        void set(const Memory& memory, int index) override;
        void set(const void *bytes, size_t bytes_length, int index) override;
        void set(const dehancer::StreamSpace& p, int index) override;

        [[nodiscard]] size_t get_block_max_size() const override;
        [[nodiscard]] ComputeSize ask_compute_size(size_t width, size_t height, size_t depth) const override;

        protected:
        dehancer::metal::Command* command_queue_ = nullptr;
        void* pipeline_ = nullptr;
        void* command_encoder_ = nullptr;
    };
}
