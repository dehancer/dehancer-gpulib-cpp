//
// Created by denn on 30.03.2022.
//

#ifndef DEHANCER_GPULIB_TEXTURESETTINGS_H
#define DEHANCER_GPULIB_TEXTURESETTINGS_H

#if defined(IOS_SYSTEM)
#import <opencv2/imgcodecs/ios.h>
#import <Metal/Metal.h>
#import <CoreImage/CoreImage.h>
#elif defined(__APPLE__)
#if defined(DEHANCER_USE_NATIVE_APPLE_API)
#defined SUPPORT_NSIMAGE 1
#import <opencv2/imgcodecs/macosx.h>
#endif
#endif

namespace dehancer::impl {
    static const CGColorSpaceRef color_space = CGColorSpaceCreateWithName(
            kCGColorSpaceSRGB
//            kCGColorSpaceDisplayP3
//            kCGColorSpaceGenericRGB
//            kCGColorSpaceITUR_709
//            kCGColorSpaceGenericRGBLinear
//            kCGColorSpaceGenericGrayGamma2_2
//            kCGColorSpaceLinearSRGB
    );
}

#endif //DEHANCER_GPULIB_TEXTURESETTINGS_H
