//
// Created by denn nevera on 10/11/2020.
//

#include "platforms/PlatformConfig.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "platforms/metal/Function.h"
#include "src/platforms/metal/Command.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "platforms/opencl/Function.h"
#include "src/platforms/opencl/Command.h"
#endif


namespace dehancer {

    namespace impl {

        class Function: public dehancer::DEHANCER_GPU_PLATFORM::Function {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Function::Function;
        };

        class Command: public dehancer::DEHANCER_GPU_PLATFORM::Command {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Command::Command;
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
