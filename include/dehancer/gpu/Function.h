//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Command.h"
#include "dehancer/gpu/CommandEncoder.h"

namespace dehancer {

    namespace impl {
        class Function;
    }

    /***
     * Function defined in kernel library that can be executed in device command queue.
     * Function has name and list of parameters can be encoded by CommadEncoder object.
     */
    class Function: public Command {
    public:

        struct ArgInfo {
            std::string name;
            uint        index;
            std::string type_name;
        };

        typedef std::function<CommandEncoder::Size (CommandEncoder& compute_encoder)> FunctionHandler;

        /***
         * Create GPU function based on kernel sourcecode. @see OpenCL C Language or Metal Shading Language
         * @param command_queue - platform based queue handler
         * @param kernel_name - kernel name defined in platform specific sourcecode
         * @param wait_until_completed - flag defines completion state
         * @param library_path - explicit shaders library file path, resource name or source bundle
         *                      (opencl source can by name of embedded value)
         *
         * @brief
         *  If wait_until_completed is set on true kernel should lock the current thread and wait when computation finish.
         *  Otherwise host code pass the next operation without locking the current thread.
         *  In this case execution result can be obtained asynchronously.
         */
        Function(const void *command_queue,
                 const std::string &kernel_name,
                 bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                 const std::string &library_path = "");

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

        /***
         * To debug current Function properties you can get Function name
         * @return function/kernel name
         */
        [[nodiscard]] const std::string& get_name() const;

        /***
         * Get Function info of argument list. ArgInfo is a structure describes arguments can be bound host and
         * device context
         * @return arg info list
         */
        [[nodiscard]] const std::vector<ArgInfo> & get_arg_list() const ;

    protected:
        std::shared_ptr<impl::Function> impl_;
    };
}


