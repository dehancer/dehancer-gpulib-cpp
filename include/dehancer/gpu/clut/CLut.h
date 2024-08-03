//
// Created by denn nevera on 2019-07-23.
//

#pragma once

#include "dehancer/gpu/Texture.h"

namespace dehancer {

    /**
     * CLut protocol
     */
    class CLut {

    public:

        static size_t default_lut_size;

        enum class Type : int {
            lut_1d   = 0, // unsupported by ofx plugin
            lut_2d   = 1,
            lut_3d   = 2,
            /// TODO to:
            lut_hald = 3
        };

        struct Options {
            size_t size = default_lut_size;
            TextureDesc::PixelFormat pixel_format = TextureDesc::PixelFormat::rgba32float;
        };

        virtual const Texture& get_texture() = 0;
        [[nodiscard]] virtual const Texture& get_texture() const = 0;
        [[nodiscard]] virtual size_t get_lut_size() const = 0;
        [[nodiscard]] virtual Type get_lut_type() const = 0;

        [[nodiscard]] virtual size_t get_channels() {
          return get_texture() ? get_texture()->get_channels() : 0;
        };

        virtual ~CLut() = default;
    };

}
