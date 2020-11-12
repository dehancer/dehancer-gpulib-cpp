//
// Created by denn nevera on 12/11/2020.
//

#include "dehancer/gpu/TextureInput.h"
#include "opencl/OCLTextureInput.h"

namespace dehancer {

    namespace impl {
        class TextureInput: public dehancer::opencl::TextureInput {
        public:
            using dehancer::opencl::TextureInput::TextureInput;
        };
    }


    TextureInput::TextureInput(const void *command_queue, const StreamSpace &space, StreamSpace::Direction direction):
    impl_(std::make_shared<impl::TextureInput>(command_queue,space,direction))
    {

    }

    Texture TextureInput::get_texture() {
      return impl_->get_texture();
    }

    const Texture TextureInput::get_texture() const {
      return impl_->get_texture();
    }

//    size_t TextureInput::get_width() const {
//      return 0;
//    }
//
//    size_t TextureInput::get_height() const {
//      return 0;
//    }
//
//    size_t TextureInput::get_depth() const {
//      return 0;
//    }
//
//    size_t TextureInput::get_channels() const {
//      return 0;
//    }
//
//    size_t TextureInput::get_length() const {
//      return 0;
//    }

    Error TextureInput::load_from_image(const std::vector<uint8_t> &buffer) {
      return impl_->load_from_image(buffer);
    }

    Error TextureInput::load_from_data(const std::vector<float> &buffer, size_t width, size_t height, size_t depth,
                                       size_t channels) {
      return impl_->load_from_data(buffer,width,height,depth,channels);
    }

    Error
    TextureInput::load_from_data(const float *buffer, size_t width, size_t height, size_t depth, size_t channels) {
      return impl_->load_from_data(buffer,width,height,depth,channels);
    }

    std::istream &operator>>(std::istream &is, TextureInput &dt) {
      if (dt.impl_)
        is>>*dt.impl_;
      return is;
    }

    TextureInput::~TextureInput() = default;

}