//
// Created by dmitry on 14.07.2025.
//

#ifndef DEHANCER_GPULIB_WAVEFORM_IMAGE_KERNEL_H
#define DEHANCER_GPULIB_WAVEFORM_IMAGE_KERNEL_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/histogram_common.h"
#include "dehancer/gpu/kernels/waveform_common.h"

DHCR_KERNEL void kernel_waveform_image(
    texture2d_read_t img DHCR_BIND_TEXTURE(0),
    DHCR_CONST_ARG int_ref_t luma_type DHCR_BIND_BUFFER(1),
    DHCR_DEVICE_ARG float4 *waveform DHCR_BIND_BUFFER(2)
    DHCR_KERNEL_GID_2D
) {
    constexpr uint numSlices = DEHANCER_WAVEFORM_SIZE; // Number of waveform points
    Texel2d tex;
    get_kernel_texel2d(img, tex);

    uint textureWidth = tex.size.x;
    uint textureHeight = tex.size.y;

    if (tex.gid.x >= numSlices) {
        return;
    }

    float sliceWidth = float(textureWidth) / float(numSlices);

    uint startCol = uint(float(tex.gid.x) * sliceWidth);
    uint endCol = uint(float(tex.gid.x + 1) * sliceWidth);

    float4 total(0.0, 0.0, 0.0, 0.0); // luma stored in w
    uint pixelCount = 0;

    for (uint row = 0; row < textureHeight; ++row) {
#pragma unroll
        for (uint col = startCol; col < endCol && col < textureWidth; ++col) {
            float4 pixel = read_image(img, make_int2(col, row));
            total.rgb += pixel.rgb;

            float luma = 0.0f;

            switch (luma_type) {
                case DEHANCER_LUMA_TYPE_YCbCr:
                    luma = dot(pixel.rgb, kIMP_Y_YCbCr_factor);
                    break;

                case DEHANCER_LUMA_TYPE_YUV:
                    luma = dot(pixel.rgb, kIMP_Y_YUV_factor);
                    break;

                case DEHANCER_LUMA_TYPE_Mean:
                    luma = dot(pixel.rgb, kIMP_Y_mean_factor);
                    break;

                default:
                    break;
            }
            total.w += luma;
            ++pixelCount;
        }
    }

    waveform[tex.gid.x] = (pixelCount > 0) ? total / float(pixelCount) : 0.0;;
}

#endif //DEHANCER_GPULIB_WAVEFORM_IMAGE_KERNEL_H
