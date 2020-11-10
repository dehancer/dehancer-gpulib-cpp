//
// Created by denn nevera on 10/11/2020.
//

#include "OCLTexture.h"
#include <cstring>

namespace dehancer::opencl {

    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc):
            OCLContext(command_queue),
            desc_(desc),
            memobj_(nullptr)
    {
      cl_image_format format;
      cl_image_desc   image_desc;

      memset( &format, 0, sizeof( format ) );

      format.image_channel_order = CL_RGBA;

      switch (desc_.pixel_format) {

        case TextureDesc::PixelFormat::rgba32float:
          format.image_channel_data_type = CL_FLOAT;
          break;

        case TextureDesc::PixelFormat::rgba16float:
          format.image_channel_data_type = CL_HALF_FLOAT;
          break;

        case TextureDesc::PixelFormat::rgba32uint:
          format.image_channel_data_type = CL_UNSIGNED_INT32;
          break;

        case TextureDesc::PixelFormat::rgba16uint:
          format.image_channel_data_type = CL_UNSIGNED_INT16;
          break;

        case TextureDesc::PixelFormat::rgba8uint:
          format.image_channel_data_type = CL_UNSIGNED_INT8;
          break;

      }
      memset( &image_desc, 0, sizeof( image_desc ) );

      switch (desc_.type) {
        case TextureDesc::Type::i1d:
          image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
          break;
        case TextureDesc::Type::i2d:
          image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
          break;
        case TextureDesc::Type::i3d:
          image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
          break;
      }

      image_desc.image_width = desc_.width;
      image_desc.image_height = desc_.height;
      image_desc.image_depth = 1;

      memobj_ = clCreateImage(
              get_context(),
              CL_MEM_READ_WRITE,
              &format,
              &image_desc,
              nullptr,
              &last_error_);
    }

    const void *TextureHolder::get_contents() const {
      return memobj_;
    }

    void *TextureHolder::get_contents() {
      return memobj_;
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
      return TextureDesc::PixelFormat::rgba8uint;
    }

    TextureDesc::Type TextureHolder::get_type() const {
      return TextureDesc::Type::i3d;
    }

    TextureHolder::~TextureHolder() {
      if(memobj_)
        clReleaseMemObject(memobj_);
    }
}