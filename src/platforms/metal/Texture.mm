//
// Created by denn nevera on 10/11/2020.
//

#include "Texture.h"
#include <cstring>

namespace dehancer::metal {

    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc, void *from_memory) :
            Context(command_queue),
            desc_(desc)
    {


      switch (desc_.pixel_format) {

        case TextureDesc::PixelFormat::rgba32float:
          break;

        case TextureDesc::PixelFormat::rgba16float:
          break;

        case TextureDesc::PixelFormat::rgba32uint:
          break;

        case TextureDesc::PixelFormat::rgba16uint:
          break;

        case TextureDesc::PixelFormat::rgba8uint:
          break;

      }

      switch (desc_.type) {
        case TextureDesc::Type::i1d:
          break;
        case TextureDesc::Type::i2d:
          break;
        case TextureDesc::Type::i3d:
          break;
      }

      unsigned char* buffer = nullptr;

      if (from_memory) {
        buffer = reinterpret_cast<unsigned char *>(from_memory);
      }


        //throw std::runtime_error("Unable to create texture: " + std::to_string(last_error_));
    }

    const void *TextureHolder::get_contents() const {
      return nullptr;
    }

    void *TextureHolder::get_contents() {
      return nullptr;
    }

    size_t TextureHolder::get_width() const {
      return desc_.width;
    }

    size_t TextureHolder::get_height() const {
      return desc_.height;
    }

    size_t TextureHolder::get_depth() const {
      return desc_.depth;
    }

    size_t TextureHolder::get_channels() const {
      return 4;
    }

    size_t TextureHolder::get_length() const {

      size_t size = desc_.width*desc_.depth*desc_.height*get_channels();

      switch (desc_.pixel_format) {

        case TextureDesc::PixelFormat::rgba32float:
          return size * sizeof(float);

        case TextureDesc::PixelFormat::rgba16float:
          return size * sizeof(float)/2;

        case TextureDesc::PixelFormat::rgba32uint:
          return size * sizeof(uint32_t);

        case TextureDesc::PixelFormat::rgba16uint:
          return size * sizeof(uint16_t);

        case TextureDesc::PixelFormat::rgba8uint:
          return size * sizeof(uint8_t);
      }
    }

    TextureDesc::PixelFormat TextureHolder::get_pixel_format() const {
      return desc_.pixel_format;
    }

    TextureDesc::Type TextureHolder::get_type() const {
      return desc_.type;
    }

    TextureHolder::~TextureHolder() {

    }
}