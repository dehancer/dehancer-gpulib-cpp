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

    /***
     * Texture output helper
     */
    class TextureOutput: public TextureIO {

    public:
        /***
         * Create texture output from command queue
         * @param command_queue  - device command queue
         * @param source - source texture
         * @param options - output options
         */
        TextureOutput(const void *command_queue,
                      const Texture& source,
                      const Options& options = {
                              .type =  Options::Type::png,
                              .compression = 0.0f
                      });

        TextureOutput(const void *command_queue,
                      size_t width, size_t height,
                      const float* from_memory = nullptr,
                      const TextureIO::Options& options = {
                              .type =  TextureIO::Options::Type::png,
                              .compression = 0.0f
                      });

        /***
         * Get texture object
         * @return texture
         */
        Texture get_texture() override;
        [[nodiscard]] const Texture get_texture() const override;

        /***
         * Write texture object to image buffer with codec defined in options TextureOutput object
         * @param buffer - image buffer
         * @return Error or Error:OK
         */
        Error write_as_image(std::vector<uint8_t>& buffer);

        /***
         * Write texture contents as raw rgba32float packed data
         * @param buffer - output buffer
         * @return Error or Error:OK
         */
        Error write_to_data(std::vector<float>& buffer);

        /***
         * Save as image into output stream
         * @param os - output stream
         * @param texture_output - texture output heler
         * @return output stream
         */
        friend std::ostream& operator<<(std::ostream& os, const TextureOutput& texture_output);

        ~TextureOutput();

    private:
        std::shared_ptr<impl::TextureOutput> impl_;
    };
}
