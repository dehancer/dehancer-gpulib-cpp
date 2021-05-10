//
// Created by denn nevera on 08/06/2020.
//

#pragma once

#include <vector>
#include <iostream>
#include "dehancer/gpu/clut/CLut.h"

namespace dehancer {

    class CubeParser {
    public:
        friend std::istream& operator>>(std::istream& is, CubeParser& dt);

        std::vector<float>& get_lut() { return buffer_; };
        [[nodiscard]] const std::vector<float>& get_lut() const { return buffer_; };
        [[nodiscard]] uint get_lut_size() const { return lut_size_; };
        [[nodiscard]] CLut::Type get_lut_type() const { return CLut::Type::lut_3d; };
        [[nodiscard]] size_t get_channels() const { return channels_; };
        [[nodiscard]] const float* get_domain_min() const { return domain_min_; };
        [[nodiscard]] const float* get_domain_max() const { return domain_max_; };

    private:
        std::vector<float> buffer_;
        size_t             lut_size_  = 0;
        size_t             channels_ = 4;
        float domain_min_[3] = { 0.0f, 0.0f, 0.0f };
        float domain_max_[3] = { 1.0f, 1.0f, 1.0f };
    };
}

