//
// Created by denn nevera on 14/11/2020.
//

#include "dehancer/gpu/TextureOutput.h"
#include "platforms/TextureOutput.h"

namespace dehancer{


    TextureOutput::TextureOutput(const void *command_queue,
                                 const Texture &source,
                                 const TextureIO::Options &options):
            TextureIO(),
            impl_(std::make_shared<impl::TextureOutput>(command_queue,source,options))
    {
    }

    TextureOutput::TextureOutput(const void *command_queue,
                                 size_t width,
                                 size_t height,
                                 const float *from_memory,
                                 const TextureIO::Options &options):
            TextureIO(),
            impl_(std::make_shared<impl::TextureOutput>(command_queue,width,height,from_memory,options)) {

    }

    const Texture& TextureOutput::get_texture() {
      return impl_->get_texture();
    }

    const Texture & TextureOutput::get_texture() const {
      return impl_->get_texture();
    }

    Error TextureOutput::write_as_image(std::vector<uint8_t> &buffer) {
      return impl_->write_as_image(buffer);
    }

    Error TextureOutput::write_to_data(std::vector<float> &buffer) {
      return impl_->write_to_data(buffer);
    }
    
    Error TextureOutput::write_as_native_image (void** handle) {
      return impl_->write_as_native_image(handle);
    }
    
    std::ostream &operator<<(std::ostream &os, const TextureOutput &dt) {
      if (dt.impl_)
        os<<*dt.impl_;
      return os;
    }
    
    TextureOutput::TextureOutput (const void *command_queue, size_t width, size_t height,
                                  const TextureIO::Options &options):TextureOutput(command_queue,width,height, nullptr, options) {
      
    }
    
    TextureOutput::~TextureOutput() = default;
}