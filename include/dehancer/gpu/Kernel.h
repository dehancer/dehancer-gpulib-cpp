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
         * @param library_path - explicit shaders library file path, resource name or source bundle
         *                      (opencl source can by name of embedded value)
         */
        explicit Kernel(
                const void *command_queue,
                const std::string& kernel_name,
                const Texture& source = nullptr,
                const Texture& destination = nullptr,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                const std::string &library_path=""
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
         * Set new source
         *
         */
        virtual void set_source(const Texture& source);
    
        /***
         * Set new destination texture
         * @param dest - texture object
         */
        virtual void set_destination(const Texture& destination);

        [[nodiscard]] virtual CommandEncoder::Size get_encoder_size() const;

        ~Kernel() override;

    private:
        std::shared_ptr<impl::Kernel> impl_;
    };

}


