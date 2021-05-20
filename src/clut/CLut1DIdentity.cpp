//
// Created by denn nevera on 22/05/2020.
//

#include "dehancer/gpu/clut/CLut1DIdentity.h"

namespace dehancer {

    CLut1DIdentity::CLut1DIdentity(
            const void *command_queue,
            size_t lut_size,
            bool wait_until_completed,
            const std::string &library_path):
            Function(command_queue, "kernel_make1DLut", wait_until_completed, library_path),
            CLut(),
            lut_size_(lut_size)
    {

        texture_ = make_texture(lut_size, lut_size, lut_size);

        execute([this](CommandEncoder& compute_encoder) {
            compute_encoder.set(texture_,0);
            compute_encoder.set(float2({1,0}),1);
            return CommandEncoder::Size::From(texture_);
        });
    }

    CLut1DIdentity::~CLut1DIdentity() = default;
}