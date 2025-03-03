//
// Created by denn nevera on 05/06/2020.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/Texture.h"

namespace dehancer {

    /***
     * Texture Input/Output interface
     */
    class TextureIO {
    public:

        TextureIO() = default;

        struct Options {
            enum Type {
                jpeg,
                png,
                tiff,
                ppm,
                bmp,
                webp
            };

            Type type = Options::Type::png;
            float compression = 0.0f;
        };

    public:

        virtual const Texture& get_texture() = 0;
        [[nodiscard]] virtual const Texture& get_texture() const = 0;

        inline static std::string extension_for(Options::Type type) {
          switch (type) {
            case Options::Type::jpeg:
              return ".jpg";
            case Options::Type::png:
              return ".png";
            case Options::Type::ppm:
              return ".ppm";
            case Options::Type::bmp:
              return ".bmp";
            case Options::Type::tiff:
              return ".tif";
            case Options::Type::webp:
              return ".webp";
          }
        }
    };
}