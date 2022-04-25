//
// Created by denn on 28.03.2022.
//

#include "platforms/TextureOutput.h"
#include <opencv2/opencv.hpp>

#if defined(IOS_SYSTEM)

#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <CoreImage/CoreImage.h>
#import <opencv2/imgcodecs/ios.h>
typedef UIImage DImage;

#elif defined(__APPLE__)

#if defined(DEHANCER_USE_NATIVE_APPLE_API)

#define SUPPORT_NSIMAGE 1

#import <Metal/Metal.h>
#import <CoreImage/CoreImage.h>
#import <AppKit/AppKit.h>

typedef NSImage DImage;

#endif
#endif

#include "TextureSettings.h"

namespace dehancer::impl {
    
    Error TextureOutput::write_as_native_image (void** handle) {
      try {
  
        id<MTLTexture> texture = reinterpret_cast<id<MTLTexture> >((__bridge id)source_->get_memory());

        NSDictionary* options = @{
                kCIImageColorSpace: (__bridge id)color_space,
                kCIContextOutputPremultiplied: @YES,
                kCIContextUseSoftwareRenderer: @FALSE
        };
        
        CGSize size = { static_cast<CGFloat>([texture width]),
                        static_cast<CGFloat>([texture height])};
        
        CIImage* ciimage = [[CIImage alloc] initWithMTLTexture:texture options:options];
  
        auto height = ciimage.extent.size.height;
          
        ciimage = [ciimage imageByApplyingTransform:CGAffineTransformMakeScale(1, -1)];
        ciimage = [ciimage imageByApplyingTransform:CGAffineTransformMakeTranslation(0, height)];
          
        if (handle) {
          CIContext *context = [CIContext contextWithOptions:nil];
          #if defined(SUPPORT_NSIMAGE)
          NSCIImageRep *rep = [NSCIImageRep imageRepWithCIImage:ciimage];
          NSImage *uiImage = [[NSImage alloc] initWithSize:rep.size];
          [uiImage addRepresentation:rep];
          #else
          CGImageRef cgImage = [context createCGImage:ciimage fromRect:[ciimage extent]];
          DImage* uiImage = [DImage imageWithCGImage:cgImage];
          CGImageRelease(cgImage);
          #endif
          
          *handle = uiImage;
        }
        else
          return Error(CommonError::EXCEPTION, "UIImage null handle");

        
        return Error(CommonError::OK);
      }
      catch (const cv::Exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
      catch (const std::exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
      
    }
}
