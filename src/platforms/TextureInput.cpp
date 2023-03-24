//
// Created by denn nevera on 12/11/2020.
//

#include "TextureInput.h"
#include <opencv2/opencv.hpp>

namespace dehancer::impl {

    Error load_from_image_data_to_buffer(const std::vector<uint8_t> &buffer,
                                         TextureDesc::PixelFormat pixelFormat_,
                                         cv::Mat& image
                                         ) {
      try {
        auto scale = 1.0f;
    
        switch (image.depth()) {
          case CV_8S:
          case CV_8U:
            switch (pixelFormat_) {
              case TextureDesc::PixelFormat::rgba8uint:
                scale = 1.0f;
                break;
              case TextureDesc::PixelFormat::rgba16uint:
                scale = 256.0f;
                break;
              default:
                scale = 1.0f/256.0f;
                break;
            }
            break;
          case CV_16U:
            switch (pixelFormat_) {
              case TextureDesc::PixelFormat::rgba8uint:
                scale = 1.0f/256.0f;
                break;
              case TextureDesc::PixelFormat::rgba16uint:
                scale = 1.0f;
                break;
              default:
                scale = 1.0f/65536.0f;
                break;
            }
            break;
          case CV_32S:
            switch (pixelFormat_) {
              case TextureDesc::PixelFormat::rgba8uint:
                scale = 256.0f/16777216.0f;
                break;
              case TextureDesc::PixelFormat::rgba16uint:
                scale = 65536.0f/16777216.0f;
                break;
              default:
                scale = 1.0f/16777216.0f;
                break;
            }
            break;
          case CV_16F:
          case CV_32F:
          case CV_64F:
            switch (pixelFormat_) {
              case TextureDesc::PixelFormat::rgba8uint:
                scale = 256.0f;
                break;
              case TextureDesc::PixelFormat::rgba16uint:
                scale = 65536.0f;
                break;
              default:
                scale = 1.0f;
                break;
            }
            break;
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
    
        switch (pixelFormat_) {
          case TextureDesc::PixelFormat::rgba8uint:
            image.convertTo(image, CV_8UC4, scale);
            break;
          case TextureDesc::PixelFormat::rgba16uint:
            image.convertTo(image, CV_16UC4, scale);
            break;
          default:
            image.convertTo(image, CV_32FC4, scale);
            break;
        }
  
        return Error(CommonError::OK);
      }
      catch (const cv::Exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
      catch (const std::exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
    }
    
    TextureInput::TextureInput(const void *command_queue, TextureDesc::PixelFormat pixelFormat):
            Command(command_queue, true),
            texture_(nullptr),
            pixelFormat_(pixelFormat)
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
    
    Error TextureInput::image_to_data (const std::vector<uint8_t> &image,
                                       TextureDesc::PixelFormat pixel_format,
                                       std::vector<uint8_t> &result,
                                       size_t& width,
                                       size_t& height,
                                       size_t& channels
                                       ) {
      width = height = channels = 0;
      auto mat = cv::Mat(cv::imdecode(image,cv::IMREAD_UNCHANGED));
      auto error = load_from_image_data_to_buffer(image, pixel_format, mat);
      if (error) { return error; }
      uint8_t *arr = mat.isContinuous() ? mat.data: mat.clone().data;
      uint length = mat.total()*mat.channels();
      result = std::vector<uint8_t>(arr, arr + length);
      width = mat.cols;
      height = mat.rows;
      channels = mat.channels();
      return Error(CommonError::OK);
    }
    
    Error TextureInput::image_to_data (const std::vector<uint8_t> &image,
                                       std::vector<uint8_t> &result,
                                       size_t& width,
                                       size_t& height,
                                       size_t& channels) {
      return image_to_data(image, pixelFormat_, result, width, height, channels);
    }
    
    
    #if not (defined(IOS_SYSTEM) and defined(DEHANCER_IOS_LOAD_NATIVE_IMAGE_LUT))
    
    Error TextureInput::load_from_image(const std::vector<uint8_t> &buffer) {

      try {
  
        auto mat = cv::Mat(cv::imdecode(buffer,cv::IMREAD_UNCHANGED));

        auto error = load_from_image_data_to_buffer(buffer, pixelFormat_, mat);

        if (error) { return error; }

        return load_from_data(reinterpret_cast<float *>(mat.data),
                              static_cast<size_t>(mat.cols),
                              static_cast<size_t>(mat.rows),
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
                .pixel_format = pixelFormat_,//TextureDesc::PixelFormat::rgba32float,
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
