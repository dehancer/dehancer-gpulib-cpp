//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_CONSTANTS_H
#define DEHANCER_GPULIB_CONSTANTS_H

#include "dehancer/gpu/kernels/types.h"

// sRGB luminance(Y) values
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_YUV_factor = {
    0.2125, 0.7154, 0.0721
};

// YCbCr luminance(Y) values
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_YCbCr_factor = {0.299, 0.587, 0.114};

// average
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_mean_factor = {0.3333, 0.3333, 0.3333};

static __constant DHCR_DEVICE_FUNC float kIMP_Std_Gamma      = 2.2;
static __constant DHCR_DEVICE_FUNC float kIMP_RGB2SRGB_Gamma = 2.4;

static __constant DHCR_DEVICE_FUNC float kIMP_Cielab_X = 95.047;
static __constant DHCR_DEVICE_FUNC float kIMP_Cielab_Y = 100.000;
static __constant DHCR_DEVICE_FUNC float kIMP_Cielab_Z = 108.883;


static __constant DHCR_DEVICE_FUNC float4 kIMP_HSV_K0      = {0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0};
static __constant DHCR_DEVICE_FUNC float4 kIMP_HSV_K1      = {0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Reds        = {315.0, 345.0, 15.0,   45.0};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Yellows     = { 15.0,  45.0, 75.0,  105.0};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Greens      = { 75.0, 105.0, 135.0, 165.0};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Cyans       = {135.0, 165.0, 195.0, 225.0};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Blues       = {195.0, 225.0, 255.0, 285.0};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Magentas    = {255.0, 285.0, 315.0, 345.0};

static __constant DHCR_DEVICE_FUNC float kIMP_COLOR_TEMP = 5000.0;
static __constant DHCR_DEVICE_FUNC float kIMP_COLOR_TINT = 0.0;

#endif //DEHANCER_GPULIB_CONSTANTS_H
