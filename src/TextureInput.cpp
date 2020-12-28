//
// Created by denn nevera on 12/11/2020.
//

#include "dehancer/gpu/TextureInput.h"
#include "platforms/TextureInput.h"

namespace dehancer {

    TextureInput::TextureInput(const void *command_queue, const StreamSpace &space, StreamSpace::Direction direction):
    TextureIO(),
    impl_(std::make_shared<impl::TextureInput>(command_queue,space,direction))
    {}

    Texture TextureInput::get_texture() {
      return impl_->get_texture();
    }

    const Texture TextureInput::get_texture() const {
      return impl_->get_texture();
    }

    Error TextureInput::load_from_image(const std::vector<uint8_t> &buffer) {
      return impl_->load_from_image(buffer);
    }

    Error TextureInput::load_from_data(const std::vector<float> &buffer, size_t width, size_t height, size_t depth) {
      return impl_->load_from_data(buffer,width,height,depth);
    }

    Error TextureInput::load_from_data(float *buffer, size_t width, size_t height, size_t depth) {
      return impl_->load_from_data(buffer,width,height,depth);
    }

    std::istream &operator>>(std::istream &is, TextureInput &dt) {
      if (dt.impl_)
        is>>*dt.impl_;
      return is;
    }

    TextureInput::~TextureInput() = default;

}