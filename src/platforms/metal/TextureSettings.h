//
// Created by denn on 30.03.2022.
//

#ifndef DEHANCER_GPULIB_TEXTURESETTINGS_H
#define DEHANCER_GPULIB_TEXTURESETTINGS_H

#if defined(IOS_SYSTEM)

#import <Metal/Metal.h>
#import <CoreImage/CoreImage.h>
#include <CoreGraphics/CoreGraphics.h>

#elif defined(__APPLE__)

#if defined(DEHANCER_USE_NATIVE_APPLE_API)
#define SUPPORT_NSIMAGE 1
#import <CoreImage/CoreImage.h>
#include <CoreGraphics/CoreGraphics.h>
#endif

#endif



namespace dehancer::impl {
    static const CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
}

#endif //DEHANCER_GPULIB_TEXTURESETTINGS_H
