//
// Created by denn nevera on 2019-07-23.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/TextureIO.h"
#include "dehancer/gpu/StreamSpace.h"
#include <vector>
#include <cinttypes>
#include <iostream>

namespace dehancer {

    namespace impl { class TextureInput; }

    class TextureInput: public TextureIO {
    public:
        explicit TextureInput(const void *command_queue,
                              const StreamSpace &space = StreamSpace::create_identity(),
                              StreamSpace::Direction direction = StreamSpace::Direction::none);

        Texture get_texture() override;
        [[nodiscard]] const Texture get_texture() const override;

        Error load_from_image(const std::vector<uint8_t>& buffer);

        Error
        load_from_data(
                const std::vector<float> &buffer,
                size_t width,
                size_t height,
                size_t depth= 1);

        Error
        load_from_data(
                float *buffer,
                size_t width,
                size_t height,
                size_t depth= 1);

        friend std::istream& operator>>(std::istream& is, TextureInput& dt);

        ~TextureInput()  ;

    private:
        std::shared_ptr<impl::TextureInput> impl_;
    };
}