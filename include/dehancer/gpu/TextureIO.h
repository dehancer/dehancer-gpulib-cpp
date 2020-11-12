//
// Created by denn nevera on 05/06/2020.
//

#pragma once


namespace dehancer {

    class TextureIO {
    public:

        struct Options {
            enum Type {
                jpeg,
                png,
                tiff
            };

            Type type;
        };

    public:

        virtual Texture get_texture() = 0;
        [[nodiscard]] virtual const Texture get_texture() const = 0;

        [[nodiscard]] virtual size_t get_width() const = 0;
        [[nodiscard]] virtual size_t get_height() const = 0;
        [[nodiscard]] virtual size_t get_depth() const = 0;
        [[nodiscard]] virtual size_t get_channels() const = 0;
        [[nodiscard]] virtual size_t get_pixel_bytes() const = 0;
    };
}