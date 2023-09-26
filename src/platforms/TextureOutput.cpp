//
// Created by denn nevera on 14/11/2020.
//

#include "TextureOutput.h"
#include  <opencv2/opencv.hpp>
#include <utility>

namespace dehancer::impl {

    TextureOutput::TextureOutput(const void *command_queue,
                                 const dehancer::Texture&  source,
                                 const dehancer::TextureIO::Options &options):
            Command(command_queue,true),
            source_(source),
            options_(options)
    {
    }

    TextureOutput::TextureOutput(const void *command_queue,
                                 size_t width,
                                 size_t height,
                                 const float* from_memory,
                                 const dehancer::TextureIO::Options &options):
            Command(command_queue,true),
            source_(nullptr),
            options_(options)
    {
      dehancer::TextureDesc desc = {
              .width = width,
              .height = height,
              .depth = 1,
              .pixel_format = TextureDesc::PixelFormat::rgba32float,
              .type = TextureDesc::Type::i2d,
              .mem_flags = TextureDesc::MemFlags::read_write
      };
      source_ = dehancer::TextureHolder::Make(command_queue, desc, from_memory);
    }

    const Texture& TextureOutput::get_texture() const {
      return source_;
    }

    const Texture& TextureOutput::get_texture() {
      return source_;
    }

    Error TextureOutput::write_as_image(std::vector<uint8_t> &buffer) const {

      std::vector<float> to_memory;
      auto ret = write_to_data(to_memory);

      if (ret) return ret;

      try {

        auto pixel_format = CV_32FC4;

        if (source_->get_desc().pixel_format == TextureDesc::PixelFormat::rgba16float) {
          pixel_format = CV_16FC4;
        }

        auto cv_result = cv::Mat(
                (int)source_->get_height(),
                (int)source_->get_width(),
                pixel_format,
                reinterpret_cast<uchar *>(to_memory.data())
        );

        std::string ext = ".png";
        std::vector<int> params;
        auto output_type = CV_8U;
        auto output_color = cv::COLOR_RGBA2BGR;
        auto scale = 255.0f;

        switch (options_.type) {

          case TextureIO::Options::Type::png:
            ext = ".png";
            params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            params.push_back((int)(9.0f * options_.compression));
            output_type = CV_16U;
            output_color = cv::COLOR_RGBA2BGRA;
            scale = 65355.0f;
            break;

          case TextureIO::Options::Type::webp:
            ext = ".webp";
            params.push_back(cv::IMWRITE_WEBP_QUALITY);
            params.push_back((int)(100.0f - 100.0f * options_.compression));
            output_type = CV_8U;
            output_color = cv::COLOR_RGBA2BGR;
            scale = 255.0f;
            break;

          case TextureIO::Options::Type::jpeg:
            ext = ".jpg";
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back((int)(100.0f - 100.0f * options_.compression));
            output_type = CV_8U;
            output_color = cv::COLOR_RGBA2BGR;
            scale = 255.0f;
            break;

          case TextureIO::Options::Type::ppm:
            ext = ".ppm";
            output_type = CV_8U;
            output_color = cv::COLOR_RGBA2BGR;
            scale = 255.0f;
            break;

          case TextureIO::Options::Type::bmp:
            ext = ".bmp";
            output_type = CV_8U;
            output_color = cv::COLOR_RGBA2BGR;
            scale = 255.0f;
            break;

          case TextureIO::Options::Type::tiff:
            ext = ".tif";
            output_type = CV_16U;
            output_color = cv::COLOR_RGBA2BGRA;
            scale = 65355.0f;
            break;
        }

        cv_result.convertTo(cv_result, output_type, scale);
        cv::cvtColor(cv_result, cv_result, output_color);

        cv::imencode(ext, cv_result, buffer, params);

        return Error(CommonError::OK);
      }
      catch (const std::exception& e) {
        return Error(CommonError::EXCEPTION, e.what());
      }
    }

    Error TextureOutput::write_to_data(std::vector<float> &buffer) const {
      return source_->get_contents(buffer);
    }

    TextureOutput::~TextureOutput() = default;

    std::ostream &operator<<(std::ostream &os, const TextureOutput &dt) {
      std::vector<uint8_t> buffer;
      auto ret = dt.write_as_image(buffer);
      if (ret) {
        throw std::runtime_error(ret.message());
      }
      os.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
      return os;
    }

#if not defined(IOS_SYSTEM)
    Error TextureOutput::write_as_native_image (void** handle) {
      return Error(CommonError::NOT_SUPPORTED);
    }
#endif
}
