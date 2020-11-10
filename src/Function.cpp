//
// Created by denn nevera on 10/11/2020.
//

//#include "dehancer/Function.h"
#include "opencl/OCLFunction.h"
#include "opencl/OCLCommand.h"

namespace dehancer {

    namespace impl {
        class Function: public dehancer::opencl::Function {
        public:
            using dehancer::opencl::Function::Function;
        };
        class Command: public dehancer::opencl::Command {
        public:
            using dehancer::opencl::Command::Command;
        };
    }

    Function::Function(const void *command_queue, const std::string &kernel_name, bool wait_until_completed):
    Command(command_queue,wait_until_completed),
    impl_(std::make_shared<impl::Function>(Command::impl_.get(), kernel_name))
    {
    }

    void Function::execute(const Function::FunctionHandler &block) {
      impl_->execute(block);
    }
}
