//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "Context.h"

namespace dehancer::metal {

    struct TextureHolder: public dehancer::TextureHolder, public Context {
        TextureHolder(const void *command_queue, const TextureDesc &desc, void *from_memory);
        ~TextureHolder() override ;

        [[nodiscard]] const void*  get_contents() const override;
        [[nodiscard]] void*  get_contents() override;
        [[nodiscard]] size_t get_width() const override;
        [[nodiscard]] size_t get_height() const override;
        [[nodiscard]] size_t get_depth() const override;
        [[nodiscard]] size_t get_channels() const override;
        [[nodiscard]] size_t get_length() const override;
        [[nodiscard]] TextureDesc::PixelFormat get_pixel_format() const override;
        [[nodiscard]] TextureDesc::Type get_type() const override;

    private:
        TextureDesc desc_;
    };
}