//
// Created by denn nevera on 17/11/2020.
//

#include "dehancer/gpu/Kernel.h"

namespace dehancer {

    namespace impl {
        struct Kernel {

            Kernel(const Texture &source,
                   const Texture &destination):
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
                   bool wait_until_completed,
                   const std::string &library_path):
            Function(command_queue, kernel_name, wait_until_completed, library_path),
            impl_(std::make_shared<impl::Kernel>(source,destination))
    {
    }

    void Kernel::process() {
      execute([this](CommandEncoder& command){
          int count = 0;
          if (this->get_source())
            command.set(this->get_source(),count++);
          if (this->get_destination())
            command.set(this->get_destination(), count++);
          this->setup(command);
          auto t = this->get_destination() ? this->get_destination() : this->get_source();
          if (t)
            return (CommandEncoder::Size){t->get_width(),t->get_height(),t->get_depth()};
          return get_encoder_size();
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
    
    void Kernel::set_source (const Texture &src) {
      impl_->source_ = src;
    }
    
    void Kernel::set_destination(const Texture &dest) {
      impl_->destination_ = dest;
    }

    CommandEncoder::Size Kernel::get_encoder_size() const {
      throw std::runtime_error("get_encoder_size must be defined for kernel: " + get_name());
    }
    
}