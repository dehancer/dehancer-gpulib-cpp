//
// Created by denn nevera on 12/11/2020.
//

#include "dehancer/gpu/TextureInput.h"
#include "platforms/TextureInput.h"

namespace dehancer {

    TextureInput::TextureInput(const void *command_queue, TextureDesc::PixelFormat pixelFormat):
    TextureIO(),
    impl_(std::make_shared<impl::TextureInput>(command_queue, pixelFormat))
    {}

    const Texture & TextureInput::get_texture() {
      return impl_->get_texture();
    }

    const Texture & TextureInput::get_texture() const {
      return impl_->get_texture();
    }
    
    Error TextureInput::load_from_image (const uint8_t *buffer, size_t length) {
      return impl_->load_from_image(std::vector<uint8_t>(buffer,buffer+length));
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
    
    Error TextureInput::image_to_data (const std::vector<uint8_t> &image,
                                       TextureDesc::PixelFormat pixel_format,
                                       std::vector<uint8_t> &result,
                                       size_t& width,
                                       size_t& height,
                                       size_t& channels) {
      return impl::TextureInput::image_to_data(image,
                                               pixel_format,
                                               result,
                                               width,
                                               height,
                                               channels
                                               );
    }
    
    Error TextureInput::image_to_data (const std::vector<uint8_t> &image,
                                       std::vector<uint8_t> &result,
                                       size_t& width,
                                       size_t& height,
                                       size_t& channels) {
      return impl_->image_to_data(image,
                                  result,
                                  width,
                                  height,
                                  channels);
    }
    
    std::istream &operator>>(std::istream &is, TextureInput &dt) {
      if (dt.impl_)
        is>>*dt.impl_;
      return is;
    }
    
    Error TextureInput::load_from_data (const std::vector<float> &buffer, size_t width, size_t height) {
      return load_from_data(buffer,width,height,1);
    }
    
    Error TextureInput::load_from_data (const std::vector<std::uint8_t> &buffer, size_t width, size_t height) {
      return impl_->load_from_data((float *) buffer.data(), width, height, 1);
    }
    
    
    Error TextureInput::load_from_native_image (const void *handle) {
      return  impl_->load_from_native_image(handle);
    }
    
    Error TextureInput::load_from_data (const std::vector<float> &buffer, size_t width) {
      return Error(CommonError::NOT_SUPPORTED);
    }
    
    Error TextureInput::load_from_data (float *buffer, size_t width) {
      return Error(CommonError::NOT_SUPPORTED);
    }
    
    
    TextureInput::~TextureInput() = default;

}