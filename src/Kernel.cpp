//
// Created by denn nevera on 17/11/2020.
//

#include "dehancer/gpu/Kernel.h"

namespace dehancer {

    namespace impl {
        struct Kernel {

            Kernel(const Texture &source, const Texture &destination):
                    source_(source),
                    destination_(destination)
            {}

            Texture source_;
            Texture destination_;
        };
    }

    Kernel::Kernel(const void *command_queue,
                   const std::string &kernel_name,
                   const Texture &source,
                   const Texture &destination,
                   bool wait_until_completed ):
            Function(command_queue,kernel_name,wait_until_completed),
            impl_(std::make_shared<impl::Kernel>(source,destination))
    {
    }

    void Kernel::process() {
      execute([this](CommandEncoder& command){
          command.set(this->get_source(),0);
          command.set(this->get_destination(), 1);
          this->setup(command);
          return this->get_destination();
      });
    }

    void Kernel::setup(CommandEncoder &commandEncoder) {
      if (optionsHandler) {
        optionsHandler(commandEncoder);
      }
    }

    Kernel::~Kernel() = default;

    const Texture& Kernel::get_source() const {
      return impl_->source_;
    }

    const Texture& Kernel::get_destination() const {
      return impl_->destination_;
    }

    void Kernel::set_destination(Texture &dest) {
      impl_->destination_ = dest;
    }
}