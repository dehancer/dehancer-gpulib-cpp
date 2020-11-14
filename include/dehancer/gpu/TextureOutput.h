//
// Created by denn nevera on 23/05/2020.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/TextureIO.h"
#include "dehancer/gpu/StreamSpace.h"
#include <vector>
#include <cinttypes>
#include <iostream>

namespace dehancer {

    namespace impl { class TextureOutput; }

    class TextureOutput: public TextureIO {

    public:
        TextureOutput(const void *command_queue,
                      const Texture& source,
                      const Options& options = {
                              .type =  Options::Type::png,
                              .compression = 0.0f
                      });

        Texture get_texture() override;
        [[nodiscard]] const Texture get_texture() const override;

        Error write_as_image(std::vector<uint8_t>& buffer);

        Error write_to_data(std::vector<float>& buffer);

        friend std::ostream& operator<<(std::ostream& os, const TextureOutput& dt);

        ~TextureOutput();

    private:
        std::shared_ptr<impl::TextureOutput> impl_;
    };
}
