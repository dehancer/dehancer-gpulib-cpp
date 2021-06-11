//
// Created by denn nevera on 2019-07-23.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLut.h"

namespace dehancer {
    
    /**
     * TODO: !!!!
     */
    class CLutHaldIdentity : public Command, public CLut {
    
    public:
        explicit CLutHaldIdentity(const void *command_queue,
                                  uint lut_size = CLut::default_lut_size,
                                  bool wait_until_completed = WAIT_UNTIL_COMPLETED
        );
        const Texture& get_texture() override { return texture_; };
        const Texture& get_texture() const override { return texture_; };
        size_t get_lut_size() const override { return lut_size_; };
        Type get_lut_type() const override { return CLut::Type::lut_hald; };
        
        ~CLutHaldIdentity() override = default;
    
    private:
        Texture  texture_;
        uint     level_;
        uint     lut_size_;
    };
}