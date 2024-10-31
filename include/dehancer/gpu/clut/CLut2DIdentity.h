//
// Created by denn nevera on 2019-07-23.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLut.h"

namespace dehancer {

    namespace impl {
        struct CLut2DIdentityImpl;
    }

    /**
     * Identity CLUT 2D function.
     */
    class CLut2DIdentity : public Function, public CLut {

    public:
        /**
         * Constructor.
         * @param command_queue Command queue to execute the gpu kernels.
         * @param lut_size Size of the lookup table.
         * @param wait_until_completed If true, wait for the kernel to complete before returning.
         * @param library_path Path to the OpenCL library.
         */
        explicit CLut2DIdentity(const void *command_queue,
                                size_t lut_size = CLut::default_lut_size,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = ""
        );

        /**
         * Constructor
         * @param command_queue - The command queue.
         * @param options      - The options for the lookup table.
         * @param wait_until_completed  - Wait until the kernel finishes.
         */
        explicit CLut2DIdentity(const void *command_queue,
                                CLut::Options options,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");

        /**
         * Get the texture of the CLUT.
         * @return
         */
        const Texture& get_texture() override;

        /**
         *  Get the texture of the CLUT.
         * @return
         */
        const Texture& get_texture() const override;

        /**
         *  Get the size of the lookup table.
         * @return
         */
        size_t get_lut_size() const override;

        /**
         *  Get the type of the lookup table.
         * @return
         */
        Type get_lut_type() const override { return Type::lut_2d; };

        ~CLut2DIdentity() override;

    private:
        std::shared_ptr<impl::CLut2DIdentityImpl> impl_;
    };
}