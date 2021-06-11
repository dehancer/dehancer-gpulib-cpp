//
// Created by denn on 26.04.2021.
//

#pragma once
#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLutTransform.h"
#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/StreamSpace.h"
#include "dehancer/gpu/TextureInput.h"

namespace dehancer {
    class CLutSquareInput: public CLut, public TextureInput{
    public:
        using TextureInput::TextureInput;
        
        explicit CLutSquareInput(const void *command_queue);
    
        Error load_from_data(float *buffer, size_t width, size_t height, size_t depth) override;
        Error load_from_data(const std::vector<float> &buffer, size_t width, size_t height, size_t depth) override;
        Error load_from_data(const std::vector<float> &buffer, size_t width, size_t height) override;
        Error load_from_image(const std::vector<uint8_t> &buffer) override;
        Error load_from_image(const uint8_t *buffer, size_t length) override;

        const Texture & get_texture() override;
        [[nodiscard]] const Texture & get_texture() const override;
        [[nodiscard]] size_t get_lut_size() const override { return lut_size_; };
        [[nodiscard]] Type get_lut_type() const override { return  CLut::Type::lut_2d; };
        
    private:
        size_t lut_size_;
        void update_lut_size();
    };
}

