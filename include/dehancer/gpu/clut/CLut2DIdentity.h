//
// Created by denn nevera on 2019-07-23.
//

#pragma once

//#include "gpulib/GpuConfig.h"
#include "dehancer/gpu/Function.h"
#include "CLut.h"

namespace dehancer {

    class CLut2DIdentity : public Function, public CLut {

    public:
        explicit CLut2DIdentity(const void *command_queue,
                                size_t lut_size = 64,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = ""
                                        );
        const Texture& get_texture() override { return texture_; };
        const Texture& get_texture() const override { return texture_; };
        size_t get_lut_size() const override { return lut_size_; };
        Type get_lut_type() const override { return Type::lut_2d; };

        ~CLut2DIdentity() override;

    private:
        Texture  texture_;
        size_t   level_;
        size_t   lut_size_;
    };
}