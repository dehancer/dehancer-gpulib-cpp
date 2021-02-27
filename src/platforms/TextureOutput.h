//
// Created by denn nevera on 14/11/2020.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/StreamSpace.h"
#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/TextureIO.h"
#include "dehancer/gpu/Command.h"

namespace dehancer::impl {

    class TextureOutput: public dehancer::Command {
    public:
        explicit TextureOutput(const void *command_queue,
                      const Texture& source,
                      const TextureIO::Options& options);

        explicit TextureOutput(const void *command_queue,
                      size_t width, size_t height,
                      const float* from_memory,
                      const TextureIO::Options& options = {
                              .type =  TextureIO::Options::Type::png,
                              .compression = 0.0f
                      });

        Texture get_texture() ;
        [[nodiscard]] const Texture get_texture() const ;

        Error write_as_image(std::vector<uint8_t>& buffer) const;

        Error write_to_data(std::vector<float>& buffer) const;

        friend std::ostream& operator<<(std::ostream& os, const TextureOutput& dt);

        ~TextureOutput() override;

    private:
        Texture source_;
        TextureIO::Options options_;
    };

}


