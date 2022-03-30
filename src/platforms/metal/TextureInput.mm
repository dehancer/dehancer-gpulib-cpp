//
// Created by denn on 28.03.2022.
//

#include "platforms/TextureInput.h"
#include <opencv2/opencv.hpp>

#if defined(IOS_SYSTEM)
#import <opencv2/imgcodecs/ios.h>
#elif defined(__APPLE__)
#if defined(DEHANCER_USE_NATIVE_APPLE_API)
#defined SUPPORT_NSIMAGE 1
#import <opencv2/imgcodecs/macosx.h>
#endif
#endif

namespace dehancer::impl {
    Error TextureInput::load_from_native_image (const void *handle) {
      try {
        cv::Mat image;
  
        ///
        /// TODO: optimization, replace opencv transformation with a native uiimage to mtltexture loader
        ///
        
        #if defined(IOS_SYSTEM)
        UIImageToMat(reinterpret_cast<UIImage*>(handle), image, true);
        #elif defined(__APPLE__) and defined(SUPPORT_NSIMAGE)
        NSImageToMat(const NSImage* image, cv::Mat& m, bool alphaExist = false);
        #else
        return Error(CommonError::NOT_SUPPORTED);
        #endif

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
}