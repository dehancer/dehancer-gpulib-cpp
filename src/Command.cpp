//
// Created by denn nevera on 09/11/2020.
//

#include "dehancer/gpu/Command.h"
#include "opencl/OCLCommand.h"

namespace dehancer {

    bool Command::WAIT_UNTIL_COMPLETED = false;

    namespace impl {
        class Command: public dehancer::opencl::Command {
        public:
            using dehancer::opencl::Command::Command;
        };
    }

    Command::Command(const void *command_queue, bool wait_until_completed):
    impl_(std::make_shared<impl::Command>(command_queue,wait_until_completed))
    {

    }

    Texture Command::make_texture(size_t width, size_t height, size_t depth) {
      return impl_->make_texture(width,height,depth);
    }

    void Command::enable_wait_completed(bool enable) {
      impl_->enable_wait_completed(enable);
    }

    bool Command::get_wait_completed() {
      return impl_->get_wait_completed();
    }

    const void *Command::get_command_queue() const {
      return impl_->get_command_queue();
    }

    void *Command::get_command_queue() {
      return impl_->get_command_queue();
    }
}
