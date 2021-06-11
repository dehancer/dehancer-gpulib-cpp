//
// Created by denn nevera on 2019-07-23.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/TextureInput.h"
#include "dehancer/gpu/clut/CLut.h"

#include <vector>
#include <cinttypes>

namespace dehancer {

    class CLutCubeInput: public CLut, protected TextureInput {
    public:
        explicit CLutCubeInput(const void *command_queue);

        const Texture& get_texture() override;
        [[nodiscard]] const Texture& get_texture() const override;
        [[nodiscard]] size_t get_lut_size() const override { return lut_size_; };
        [[nodiscard]] Type get_lut_type() const override { return Type::lut_3d; };

        Error load_from_data(float *buffer, size_t lut_size);
        Error load_from_data(const std::vector<float> &buffer, size_t lut_size);
        
        //Error load_from_data(const float* buffer, size_t size, size_t channels);
        //Error load_from_data(const std::vector<float>& buffer, size_t size, size_t channels);

        friend std::istream& operator>>(std::istream& is, CLutCubeInput& dt);

        ~CLutCubeInput() override ;

    private:
        //std::shared_ptr<TextureInput> texture_input_;
        size_t lut_size_;
    };
}