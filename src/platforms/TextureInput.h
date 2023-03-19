//
// Created by denn nevera on 12/11/2020.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/StreamSpace.h"
#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Command.h"

namespace dehancer::impl {

    class TextureInput: public dehancer::Command {

    public:

        explicit TextureInput(const void *command_queue, TextureDesc::PixelFormat pixelFormat);

        const Texture& get_texture() { return texture_; };
        [[nodiscard]] const Texture& get_texture() const { return texture_; };

        [[nodiscard]] size_t get_width() const;
        [[nodiscard]] size_t get_height() const;
        [[nodiscard]] size_t get_depth() const;
        [[nodiscard]] size_t get_channels() const;
        [[nodiscard]] size_t get_length() const;

        Error load_from_image(const std::vector<uint8_t>& buffer);
        
        Error load_from_native_image(const void* handle);

        Error
        load_from_data(
                const std::vector<float> &buffer,
                size_t width,
                size_t height,
                size_t depth = 1);

        Error
        load_from_data(
                float *buffer,
                size_t width,
                size_t height,
                size_t depth = 1);

        friend std::istream& operator>>(std::istream& is, TextureInput& dt);

        ~TextureInput() override ;

    private:
        Texture texture_;
        TextureDesc::PixelFormat pixelFormat_;
    };
}

