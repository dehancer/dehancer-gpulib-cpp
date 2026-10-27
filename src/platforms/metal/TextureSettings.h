//
// Created by denn on 30.03.2022.
//

#ifndef DEHANCER_GPULIB_TEXTURESETTINGS_H
#define DEHANCER_GPULIB_TEXTURESETTINGS_H

#if defined(IOS_SYSTEM)

#include <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <Metal/Metal.h>

#define SUPPORT_UIIMAGE 1

#elif defined(__APPLE__)

#if defined(DEHANCER_USE_NATIVE_APPLE_API)

#define SUPPORT_NSIMAGE 1

#include <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#endif

#endif

#if defined(SUPPORT_NSIMAGE) || defined(SUPPORT_UIIMAGE)
namespace dehancer::impl {
static const CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
static const CIFormat pixel_format = kCIFormatRGBAh;
}  // namespace dehancer::impl
#endif

#endif  // DEHANCER_GPULIB_TEXTURESETTINGS_H
