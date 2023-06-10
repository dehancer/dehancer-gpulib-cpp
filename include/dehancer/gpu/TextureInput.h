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

    /***
     * Texture input helper
     */
    class TextureInput: public TextureIO {
    public:
        /***
         * Create texture input from command queue
         * @param command_queue - device command queue
         * @param space - color space transformation
         * @param direction - color space transformation direction
         * @param pixel_format - pixel format
         */
        explicit TextureInput(const void *command_queue,
                              TextureDesc::PixelFormat pixel_format = TextureDesc::PixelFormat::rgba32float);

        /***
         * Get texture object
         * @return dehancer::Texture object
         */
        const Texture & get_texture() override;
        [[nodiscard]] const Texture & get_texture() const override;

        /***
         * Load texture from Image buffer. Buffer can contain data with one of defined image codec.
         * @param buffer
         * @return error or ok
         */
        virtual Error load_from_image(const std::vector<uint8_t>& buffer);
    
       /***
        *
        * Transform image format buffer to continues pixels array buffer.
        *
        * @param image
        * @param result
        * @param pixel_format
        * @return
        */
        static Error image_to_data(
                const std::vector<uint8_t>& image,
                TextureDesc::PixelFormat pixel_format,
                std::vector<uint8_t>& result,
                size_t& width,
                size_t& height,
                size_t& channels
                );
    
        /**
         * Transform image format buffer to continues pixels array buffer.
         *
         * @param image
         * @param result
         * @return
         */
        virtual Error image_to_data(
                const std::vector<uint8_t>& image,
                std::vector<uint8_t>& result,
                size_t& width,
                size_t& height,
                size_t& channels);
    
        /***
         * Load texture from Image buffer. Buffer can contain data with one of defined image codec.
         * @param buffer
         * @param length
         * @return error or ok
         */
        virtual Error load_from_image(const uint8_t* buffer, size_t length);
    
        /***
         * Load texture from native system image representation like a UIImage in iOS
         * @param handle
         * @return error or ok
         */
        virtual Error load_from_native_image(const void* handle);
    
        /***
         * Load texture raw data packed as rgba32float
         * @param buffer
         * @param width
         * @param height
         * @param depth
         * @return error or ok
         */
        virtual Error load_from_data(
                const std::vector<float> &buffer,
                size_t width,
                size_t height,
                size_t depth);
    
        virtual Error load_from_data(
                const std::vector<float> &buffer,
                size_t width,
                size_t height);
    
        virtual Error load_from_data(
                const std::vector<std::uint8_t> &buffer,
                size_t width,
                size_t height);
    
        virtual Error load_from_data(
                const std::vector<float> &buffer,
                size_t width);
    
        /***
         * Load texture raw data packed as rgba32float
         * @param buffer
         * @param width
         * @param height
         * @param depth
         * @return error or ok
         */
        virtual Error load_from_data(
                float *buffer,
                size_t width,
                size_t height,
                size_t depth);
    
        virtual Error load_from_data(
                float *buffer,
                size_t width,
                size_t height) {
          return load_from_data(buffer,width,height,1);
        };
    
        virtual Error load_from_data(
                float *buffer,
                size_t width);
        
        /***
         * Read image to the Texture from input stream
         * @param os - input stream
         * @param texture_output - texture output heler
         * @return input stream
         */
        friend std::istream& operator>>(std::istream& is, TextureInput& texture_output);

        virtual ~TextureInput()  ;

    private:
        std::shared_ptr<impl::TextureInput> impl_;
    };
}