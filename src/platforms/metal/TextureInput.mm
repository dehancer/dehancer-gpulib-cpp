//
// Created by denn on 28.03.2022.
//

#include "platforms/TextureInput.h"
#include <opencv2/opencv.hpp>

#if defined(IOS_SYSTEM)
#import <opencv2/imgcodecs/ios.h>
#import <Metal/Metal.h>
#import <CoreImage/CoreImage.h>
typedef UIImage DImage;

#define SUPPORT_UIIMAGE 1

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
    Error TextureInput::load_from_native_image (const void *handle) {
      #if defined(SUPPORT_NSIMAGE) || defined(SUPPORT_UIIMAGE)
  
      try {
  
        auto command_queue = reinterpret_cast<id<MTLCommandQueue> >((__bridge id)get_command_queue());
        id<MTLDevice>  device = command_queue.device;
        
        NSDictionary* options = @{
                kCIImageColorSpace: (__bridge id)color_space,
                kCIContextOutputPremultiplied: @YES,
                kCIContextUseSoftwareRenderer: @FALSE
        };
        
        CIContext* context = [CIContext contextWithMTLDevice: device ];
  
        #if defined(SUPPORT_NSIMAGE)
        NSData* tiffData = [reinterpret_cast<DImage*>(handle) TIFFRepresentation];
        NSBitmapImageRep* bitmap = [NSBitmapImageRep imageRepWithData:tiffData];

        CIImage* ciimage = [[CIImage alloc] initWithBitmapImageRep:bitmap];
        #else
          CIImage* ciimage = [[CIImage alloc] initWithImage: reinterpret_cast<DImage*>(handle) options: options];
        #endif
  
        auto height = ciimage.extent.size.height;
  
        ciimage = [ciimage imageByApplyingTransform:CGAffineTransformMakeScale(1, -1)];
        ciimage = [ciimage imageByApplyingTransform:CGAffineTransformMakeTranslation(0, height)];
  
        dehancer::TextureDesc desc = {
                .width = static_cast<size_t>(ciimage.extent.size.width),
                .height = static_cast<size_t>(ciimage.extent.size.height),
                .depth = 1,
                .pixel_format = TextureDesc::PixelFormat::rgba16float,
                .type = dehancer::TextureDesc::Type::i2d,
                .mem_flags = TextureDesc::MemFlags::read_write
        };
        
        texture_ = desc.make(command_queue);
        
        if (!texture_)
          return Error(CommonError::EXCEPTION, "Texture not created");
        
        auto texture = reinterpret_cast<id<MTLTexture> >((__bridge id)texture_->get_memory());
        
        id <MTLCommandBuffer> commandBuffer = [command_queue commandBuffer];
  
        [context render:ciimage
           toMTLTexture:texture
          commandBuffer:commandBuffer
                 bounds:{0, 0, static_cast<CGFloat>(desc.width), static_cast<CGFloat>(desc.height)}
             colorSpace: color_space
        ];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
  
        return Error(CommonError::OK);
      }
      catch (const cv::Exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
      catch (const std::exception & e) { return Error(CommonError::EXCEPTION, e.what()); }
      #else
      return Error(CommonError::NOT_SUPPORTED);
      #endif
    }
}