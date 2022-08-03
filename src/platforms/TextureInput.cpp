//
// Created by denn nevera on 12/11/2020.
//

#include "TextureInput.h"
#include <opencv2/opencv.hpp>

namespace dehancer::impl {

    TextureInput::TextureInput(const void *command_queue):
            Command(command_queue, true),
            texture_(nullptr)
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
    
    #if not defined(IOS_SYSTEM)
    Error TextureInput::load_from_image(const std::vector<uint8_t> &buffer) {

      try {
        auto image = cv::Mat(cv::imdecode(buffer,cv::IMREAD_UNCHANGED));

        auto scale = 1.0f;

        switch (image.depth()) {
          case CV_8S:
          case CV_8U:
            scale = 1.0f/256.0f;
            break;
          case CV_16U:
            scale = 1.0f/65536.0f;
            break;
          case CV_32S:
            scale = 1.0f/16777216.0f;
            break;
          case CV_16F:
          case CV_32F:
          case CV_64F:
            scale = 1.0f;
            break;
          default:
            return Error(CommonError::NOT_SUPPORTED, error_string("Image pixel depth is not supported"));
        }

        auto color_cvt = cv::COLOR_BGR2RGBA;
        if (image.channels() == 1){
          color_cvt = cv::COLOR_GRAY2RGBA;
        }
        else if (image.channels() == 3){
          color_cvt = cv::COLOR_BGR2RGBA;
        }
        else if (image.channels() == 4){
          color_cvt = cv::COLOR_BGRA2RGBA;
        }
        else {
          return Error(CommonError::NOT_SUPPORTED, error_string("Image channels depth is not supported"));
        }

        cv::cvtColor(image, image, color_cvt);

        image.convertTo(image, CV_32FC4, scale);

        return load_from_data(reinterpret_cast<float *>(image.data),
                              static_cast<size_t>(image.cols),
                              static_cast<size_t>(image.rows),
                              1
        );

      }
      catch (const cv::Exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
      catch (const std::exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
    }
    #endif
    
    Error TextureInput::load_from_data(const std::vector<float> &buffer, size_t width, size_t height, size_t depth) {
      auto* _buffer = const_cast<float *>(buffer.data());
      return load_from_data(_buffer, width, height, depth);
    }

    Error
    TextureInput::load_from_data(float *buffer, size_t width, size_t height, size_t depth) {
      try {
        TextureDesc::Type type = TextureDesc::Type::i2d;

        if (depth > 1) {
          type = TextureDesc::Type::i3d;
        } else if (height == 1) {
          type = TextureDesc::Type::i1d;
        }

        dehancer::TextureDesc desc = {
                .width = width,
                .height = height,
                .depth = depth,
                .pixel_format = TextureDesc::PixelFormat::rgba32float,
                .type = type,
                .mem_flags = TextureDesc::MemFlags::read_write
        };

        texture_ = TextureHolder::Make(get_command_queue(), desc, buffer);

        if (!texture_)
          return Error(CommonError::NOT_SUPPORTED,
                       error_string("Texture could not be created from the image"));
      }
      catch (const std::exception & e) { return Error(CommonError::EXCEPTION, e.what()); }

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
    
    #if not defined(__APPLE__)
    Error TextureInput::load_from_native_image (const void *handle) {
      return Error(CommonError::NOT_SUPPORTED);
    }
    #endif
    TextureInput::~TextureInput() = default;
}
