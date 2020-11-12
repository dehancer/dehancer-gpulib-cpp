//
// Created by denn nevera on 12/11/2020.
//

#include "OCLTextureInput.h"
#include <opencv4/opencv2/opencv.hpp>

namespace dehancer::opencl {
    TextureInput::TextureInput(const void *command_queue, const dehancer::StreamSpace &space,
                               dehancer::StreamSpace::Direction direction):
                               OCLContext(command_queue),
                               texture_(nullptr),
                               space_(space),
                               direction_(direction)
    {
    }

    size_t TextureInput::get_width() const {
      return texture_->get_width();
    }

    size_t TextureInput::get_height() const {
      return texture_->get_height();
    }

    size_t TextureInput::get_depth() const {
      return texture_->get_depth();
    }

    size_t TextureInput::get_channels() const {
      return texture_->get_channels();
    }

    size_t TextureInput::get_length() const {
      return texture_->get_length();
    }

    Error TextureInput::load_from_image(const std::vector<uint8_t> &buffer) {

      try {
        //auto image = cv::imdecode(buffer, cv::IMREAD_COLOR|cv::IMREAD_ANYDEPTH);
        //auto image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        //cv::cvtColor(image,image, CV_32F);
        //cv::cvtColor(image,image,cv::COLOR_BGR2RGBA);

        auto* image = new cv::Mat(cv::imdecode(buffer, cv::IMREAD_COLOR));
        cv::cvtColor(*image,*image, CV_32FC4);

        dehancer::TextureDesc desc = {
                .width = static_cast<size_t>(image->cols),
                .height = static_cast<size_t>(image->rows),
                .depth = 1,
                .pixel_format = TextureDesc::PixelFormat::rgba32float,
                .type = TextureDesc::Type::i2d,
                .mem_flags = TextureDesc::MemFlags::read_only
        };

        texture_ = TextureHolder::Make(get_command_queue(), desc, image->data);
      }
      catch (const std::exception & e) { return Error(CommonError::EXCEPTION, e.what()); }

      return Error(CommonError::OK);
    }

    Error TextureInput::load_from_data(const std::vector<float> &buffer, size_t width, size_t height, size_t depth,
                                       size_t channels) {
      return load_from_data(buffer.data(), width, height, depth, channels);
    }

    Error
    TextureInput::load_from_data(const float *buffer, size_t width, size_t height, size_t depth, size_t channels) {
      TextureDesc::Type type = TextureDesc::Type::i2d;

      if (depth>1) {
        type = TextureDesc::Type::i3d;
      }
      else if (height==1) {
        type = TextureDesc::Type::i1d;
      }

      dehancer::TextureDesc desc = {
              .width = width,
              .height = height,
              .depth = depth,
              .pixel_format = TextureDesc::PixelFormat::rgba32float,
              .type = type,
              .mem_flags = TextureDesc::MemFlags::read_only
      };

      texture_ = TextureHolder::Make(get_command_queue(), desc, (void *) buffer);
      return Error(CommonError::OK);
    }

    std::istream &operator>>(std::istream &is, TextureInput &dt) {

      std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());

      auto error = dt.load_from_image(buffer);

      if (error) {
        throw std::istream::failure(error.message());
      }

      return is;
    }

    TextureInput::~TextureInput() = default;
}
