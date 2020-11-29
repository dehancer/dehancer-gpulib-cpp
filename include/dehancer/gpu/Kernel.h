//
// Created by denn nevera on 17/11/2020.
//

#pragma once

#include "dehancer/gpu/Function.h"

namespace dehancer {

    namespace impl { struct Kernel; }

    /***
     * Kernel is a Function functor process kernel functiob over source texture and puts
     * result to destination.
     */
    class Kernel: public Function  {

    public:
        /***
         * Create Kernel object
         * @param command_queue - platform based command queue
         * @param kernel_name - kernel name in sourcecode
         * @param source - source kernel texture
         * @param destination - destination texture
         * @param wait_until_completed - flag defines completion state
         */
        explicit Kernel(
                const void *command_queue,
                const std::string& kernel_name,
                const Texture& source,
                const Texture& destination,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED
        );

        /***
         * Custom options handler
         */
        FunctionHandler optionsHandler = nullptr;

        /***
         * Process Kernel functor
         */
        virtual void process();

        /***
         * Set up kernel parameters
         * @param encode - command encoder interface
         */
        virtual void setup(CommandEncoder &encode);

        /**
         * Get source texture
         * @return texture object
         */
        [[nodiscard]] virtual const Texture& get_source() const;

        /***
         * Get destination texture
         * @return texture object
         */
        [[nodiscard]] virtual const Texture& get_destination() const;

        /***
         * Set new destination texture
         * @param dest - texture object
         */
        virtual void set_destination(Texture& dest);

        [[nodiscard]] virtual CommandEncoder::Size get_encoder_size() const;

        ~Kernel() override;

    private:
        std::shared_ptr<impl::Kernel> impl_;
    };

}


