//
// Created by denn nevera on 05/06/2020.
//

#pragma once
#include "dehancer/Common.h"
#include "dehancer/gpu/Texture.h"

namespace dehancer {

    class TextureIO /*: public TextureHolder */{
    public:

        TextureIO() = default; //: TextureHolder() {}

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
    };
}