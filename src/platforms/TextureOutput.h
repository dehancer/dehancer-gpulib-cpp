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
        TextureOutput(const void *command_queue,
                      const Texture& source,
                      const TextureIO::Options& options);

        Texture get_texture() ;
        [[nodiscard]] const Texture get_texture() const ;

        Error write_as_image(std::vector<uint8_t>& buffer) const;

        Error write_to_data(std::vector<float>& buffer) const;

        friend std::ostream& operator<<(std::ostream& os, const TextureOutput& dt);

        ~TextureOutput();

    private:
        Texture source_;
        TextureIO::Options options_;
    };

}


