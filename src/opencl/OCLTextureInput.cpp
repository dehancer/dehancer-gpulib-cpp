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
        auto stdw = std::setw(15);

        auto image = cv::Mat(cv::imdecode(buffer,cv::IMREAD_UNCHANGED));

        //std::cout << stdw << "SOURCE Pixel Depth:" <<  image.depth() << std::endl;
        //std::cout << stdw << "SOURCE Pixel Chann:" <<  image.channels() << std::endl;

        std::cout << std::left << std::setfill(' ') <<  std::setw(15) << " Read buffer " << std:: endl;

        std::cout << stdw << "Width:" <<  image.cols << std::endl;
        std::cout << stdw << "Height:" <<  image.rows << std::endl;

        std::cout << stdw << "Pixel Depth:" <<  image.depth() << std::endl;
        std::cout << stdw << "Pixel Dims:" <<  image.dims << std::endl;
        std::cout << stdw << "Channels:" <<  image.channels() << std::endl;

        std::cout << stdw << "Width Step:" <<  image.step << std::endl;
        std::cout << stdw << "Image Size:" <<  image.size << std::endl;
        std::cout << stdw << "Image Type:" <<  image.type() << std::endl;
        std::cout << stdw << "Image Flags:" << image.flags << std::endl;


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

        dehancer::TextureDesc desc = {
                .width = static_cast<size_t>(image.cols),
                .height = static_cast<size_t>(image.rows),
                .depth = 1,
                .pixel_format = TextureDesc::PixelFormat::rgba32float,
                .type = TextureDesc::Type::i2d,
                .mem_flags = TextureDesc::MemFlags::read_only
        };

        texture_ = TextureHolder::Make(get_command_queue(), desc, image.data);
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
