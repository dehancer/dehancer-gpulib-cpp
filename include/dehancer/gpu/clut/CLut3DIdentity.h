//
// Created by denn nevera on 2019-07-23.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLut.h"

namespace dehancer {

    namespace impl {
        struct CLut3DIdentityImpl;
    }
    /**
     * CLut3DIdentity represents a 3D lookup table that is an identity function.
     */
    class CLut3DIdentity : public Function, public CLut {

    public:
        /**
         * Constructor
         * @param command_queue - The command queue.
         * @param lut_size    - The size of the lookup table.
         * @param wait_until_completed  - Wait until the kernel finishes.
         * @param library_path  - The path to the OpenCL library.
         */
        explicit CLut3DIdentity(const void *command_queue,
                       size_t lut_size = CLut::default_lut_size,
                       bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                       const std::string &library_path = "");

        /**
         * Constructor
         * @param command_queue - The command queue.
         * @param options      - The options for the lookup table.
         * @param wait_until_completed  - Wait until the kernel finishes.
         */
        explicit CLut3DIdentity(const void *command_queue,
                                CLut::Options options,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");

        /**
         * Get the texture of the lookup table.
         * @return texting pointer to the texture of the lookup table.
         */
        const Texture& get_texture() override;

        /**
         * Get the texture of the lookup table.
         * @return A const reference to the texture of the lookup table.
         */
        const Texture& get_texture() const override;

        /**
         * Get the size of the lookup table.
         * @return The size of the lookup table.
         */
        size_t get_lut_size() const override;

        /**
         * Get the type of the lookup table.
         * @return The type of the lookup table.
         */
        Type get_lut_type() const override { return Type::lut_3d; };

        ~CLut3DIdentity() override ;

    private:
        std::shared_ptr<impl::CLut3DIdentityImpl> impl_;
    };
}