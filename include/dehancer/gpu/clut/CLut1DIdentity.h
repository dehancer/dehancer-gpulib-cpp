//
// Created by denn nevera on 22/05/2020.
//

#pragma once

#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/Function.h"

namespace dehancer {

    /**
     * Identity CLUT function that uses a single 1D texture for look-up.
     */
    class CLut1DIdentity : public Function, public CLut {

    public:
        /**
         *  Constructor.
         * @param command_queue
         * @param lut_size
         * @param wait_until_completed
         * @param library_path
         */
        explicit CLut1DIdentity(const void *command_queue,
                                size_t lut_size = CLut::default_lut_size,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");

        const Texture& get_texture() override { return texture_; };
        const Texture& get_texture() const override { return texture_; };
        size_t get_lut_size() const override { return lut_size_; };
        Type get_lut_type() const override { return Type::lut_3d; };

        ~CLut1DIdentity() override ;

    private:
        Texture  texture_;
        size_t    lut_size_;
    };
}