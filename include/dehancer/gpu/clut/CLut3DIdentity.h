//
// Created by denn nevera on 2019-07-23.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLut.h"

namespace dehancer {

    class CLut3DIdentity : public Function, public CLut {

    public:
        explicit CLut3DIdentity(const void *command_queue,
                       uint lut_size = 64,
                       bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                       const std::string &library_path = "");
        const Texture& get_texture() override { return texture_; };
        const Texture& get_texture() const override { return texture_; };
        size_t get_lut_size() const override { return lut_size_; };
        Type get_lut_type() const override { return Type::lut_3d; };

        ~CLut3DIdentity() override ;

    private:
        Texture  texture_;
        uint     lut_size_;
    };
}