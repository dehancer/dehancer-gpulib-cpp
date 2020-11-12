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

        //[[nodiscard]] const void*  get_contents() const override ;
        //[[nodiscard]] void*  get_contents() override ;
        //[[nodiscard]] size_t get_width() const override;
        //[[nodiscard]] size_t get_height() const override;
        //[[nodiscard]] size_t get_depth() const override;
        //[[nodiscard]] size_t get_channels() const override;
        //[[nodiscard]] size_t get_length() const override;

        Error load_from_image(const std::vector<uint8_t>& buffer);

        Error
        load_from_data(
                const std::vector<float> &buffer,
                size_t width,
                size_t height,
                size_t depth= 1,
                size_t channels= 4);

        Error
        load_from_data(
                const float *buffer,
                size_t width,
                size_t height,
                size_t depth= 1,
                size_t channels= 4);

        friend std::istream& operator>>(std::istream& is, TextureInput& dt);

        ~TextureInput()  ;

    private:
        std::shared_ptr<impl::TextureInput> impl_;
    };
}