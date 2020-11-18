//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "Command.h"

namespace dehancer {

    namespace impl {
        class Function;
    }

    /***
     * Command encoder offers methods to bind parameters that are objects
     * of host system and placed in CPU memory and GPU kernel functions.
     */
    class CommandEncoder {
    public:
        /***
         * Bind texture object with kernel argument placed at defined index. @see Texture
         * @param texture - texture object
         * @param index - index place at kernel parameter list
         */
        virtual void set(const Texture& texture, int index) = 0;

        /***
        * Bind memory object with kernel argument placed at defined index. @see Memory
        * @param texture - texture object
        * @param index - index place at kernel parameter list
        */
        virtual void set(const Memory& memory, int index) = 0;

        /***
         * Bind raw bytes with kernel argument placed at defined index. @see Texture
         * @param bytes - host memory buffer
         * @param bytes_length - buffer length
         * @param index - index place at kernel parameter list
         */
        virtual void set(const void* bytes, size_t bytes_length, int index) = 0;
    };

    /***
     * Function defined in kernel library that can be executed in device command queue.
     * Function has name and list of parameters can be encoded by CommadEncoder object.
     */
    class Function: public Command {
    public:
        typedef std::function<Texture (CommandEncoder& compute_encoder)> FunctionHandler;

        /***
         * Create GPU function based on kernel sourcecode. @see OpenCL C Language or Metal Shading Language
         * @param command_queue - platform based queue handler
         * @param kernel_name - kernel name defined in platform specific sourcecode
         * @param wait_until_completed - flag defines completion state
         *
         * @brief
         *  If wait_until_completed is set on true kernel should lock the current thread and wait when computation finish.
         *  Otherwise host code pass the next operation without locking the current thread.
         *  In this case execution result can be obtained asynchronously.
         */
        Function(const void *command_queue, const std::string& kernel_name, bool wait_until_completed = WAIT_UNTIL_COMPLETED);

        /***
         * Execute named kernel function in lambda block.
         * @param block
         *
         * @example
         * auto kernel = dehancer::Function(command_queue, "kernel");
         * auto result = kernel.make_texture(width,height);
         * kernel.execute([&result](dehancer::CommandEncoder& command_encoder){
         *      command_encoder.set(result, 0);
         *      return result;
         * });
         *
         * Block lambda must return Texture object or nullptr.
         *
         */
        void execute(const FunctionHandler& block);

    protected:
        std::shared_ptr<impl::Function> impl_;
    };
}


