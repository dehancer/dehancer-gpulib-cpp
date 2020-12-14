//
// Created by denn nevera on 10/11/2020.
//

#include "Texture.h"
#include <cstring>

namespace dehancer::opencl {

    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc, const void *from_memory) :
            dehancer::TextureHolder(),
            Context(command_queue),
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
      image_desc.image_depth = desc_.depth;

      cl_mem_flags mem_flags = 0;

      mem_flags |= TextureDesc::MemFlags::read_write & desc.mem_flags ? CL_MEM_READ_WRITE : 0;
      mem_flags |= TextureDesc::MemFlags::read_only & desc.mem_flags ? CL_MEM_READ_ONLY : 0;
      mem_flags |= TextureDesc::MemFlags::write_only & desc.mem_flags ? CL_MEM_WRITE_ONLY : 0;

      unsigned char* buffer = nullptr;

      if (from_memory) {
        buffer = reinterpret_cast<unsigned char *>((void *)from_memory);
        mem_flags |= CL_MEM_COPY_HOST_PTR;
      }

      memobj_ = clCreateImage(
              get_context(),
              mem_flags,
              &format,
              &image_desc,
              buffer,
              &last_error_);

      if (last_error_ != CL_SUCCESS)
        throw std::runtime_error("Unable to create texture: " + std::to_string(last_error_));
    }

    const void *TextureHolder::get_memory() const {
      return memobj_;
    }

    void *TextureHolder::get_memory() {
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
      return desc_.channels;
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
      if(memobj_)
        clReleaseMemObject(memobj_);
    }

    dehancer::Error TextureHolder::get_contents(std::vector<float> &buffer) const {
      buffer.resize( get_length());
      return get_contents(buffer.data(), get_length());
    }

    dehancer::Error TextureHolder::get_contents(void *buffer, size_t length) const {
      size_t originst[3];
      size_t regionst[3];
      size_t  rowPitch = 0;
      size_t  slicePitch = 0;
      originst[0] = 0; originst[1] = 0; originst[2] = 0;
      regionst[0] = get_width();
      regionst[1] = get_height();
      regionst[2] = get_depth();

      if (length< this->get_length()) {
        return Error(CommonError::OUT_OF_RANGE, "Texture length greater then buffer length");
      }

      auto ret = clEnqueueReadImage(
              get_command_queue(),
              memobj_,
              CL_TRUE,
              originst,
              regionst,
              rowPitch,
              slicePitch,
              buffer,
              0,
              nullptr,
              nullptr );

      if (ret != CL_SUCCESS) {
        return Error(CommonError::EXCEPTION, "Texture could not be read");
      }

      return Error(CommonError::OK);
    }

}