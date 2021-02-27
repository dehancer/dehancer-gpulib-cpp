//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_CONSTANTS_H
#define DEHANCER_GPULIB_CONSTANTS_H

#include "dehancer/gpu/kernels/types.h"

// sRGB luminance(Y) values
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_YUV_factor = {
    0.2125f, 0.7154f, 0.0721f
};

// YCbCr luminance(Y) values
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_YCbCr_factor = {0.299f, 0.587f, 0.114f};

// average
static __constant DHCR_DEVICE_FUNC float3 kIMP_Y_mean_factor = {0.3333f, 0.3333f, 0.3333f};

static __constant DHCR_DEVICE_FUNC float kIMP_Std_Gamma      = 2.2f;
static __constant DHCR_DEVICE_FUNC float kIMP_RGB2SRGB_Gamma = 2.4f;

static __constant DHCR_DEVICE_FUNC float kIMP_Cielab_X = 95.047f;
static __constant DHCR_DEVICE_FUNC float kIMP_Cielab_Y = 100.000f;
static __constant DHCR_DEVICE_FUNC float kIMP_Cielab_Z = 108.883f;


static __constant DHCR_DEVICE_FUNC float4 kIMP_HSV_K0      = {0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f};
static __constant DHCR_DEVICE_FUNC float4 kIMP_HSV_K1      = {0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Reds        = {315.0f, 345.0f, 15.0f,   45.0f};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Yellows     = { 15.0f,  45.0f, 75.0f,  105.0f};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Greens      = { 75.0f, 105.0f, 135.0f, 165.0f};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Cyans       = {135.0f, 165.0f, 195.0f, 225.0f};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Blues       = {195.0f, 225.0f, 255.0f, 285.0f};
static __constant DHCR_DEVICE_FUNC float4 kIMP_Magentas    = {255.0f, 285.0f, 315.0f, 345.0f};

static __constant DHCR_DEVICE_FUNC float kIMP_COLOR_TEMP = 5000.0f;
static __constant DHCR_DEVICE_FUNC float kIMP_COLOR_TINT = 0.0f;

#endif //DEHANCER_GPULIB_CONSTANTS_H
