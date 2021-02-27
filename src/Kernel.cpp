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
          if (impl_->source_)
            command.set(impl_->source_,count++);
          if (impl_->destination_)
            command.set(impl_->destination_, count++);
          this->setup(command);
          auto t  = impl_->destination_ ? impl_->destination_ : impl_->source_;
          if (t)
            return CommandEncoder::Size::From(t);
          return get_encoder_size();
      });
    }
    
    void Kernel::process (const Texture &source, const Texture &destination) {
      set_source(source);
      set_destination(destination);
      process();
    }
    
    void Kernel::setup(CommandEncoder &commandEncoder) {
      if (encode_handler) {
        encode_handler(commandEncoder);
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