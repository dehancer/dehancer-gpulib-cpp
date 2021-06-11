//
// Created by denn on 26.05.2021.
//

#pragma once

#include "dehancer/gpu/Function.h"

namespace dehancer {
    class CLut3DCopyFunction: public Function {
    public:
        
        CLut3DCopyFunction(const void *command_queue,
                           Texture const &input,
                           size_t lut_size,
                           bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                           const std::string &library_path = "");
        
        void foreach(std::function<void(uint index, float r, float g, float b)> block);
        size_t get_lut_size() const;
        size_t get_image_bytes() const;
        size_t get_bytes_per_image() const;
        
    private:
        size_t lut_size_;
        size_t image_size_;
        size_t channels_;
        std::vector<float> buffer_;
    };
}
