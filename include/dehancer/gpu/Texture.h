//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include <memory>

namespace dehancer {

    struct TextureDesc {

        enum MemFlags:uint32_t {
            read_write = (1 << 0),
            write_only = (1 << 1),
            read_only  = (1 << 2),
        };

        enum class PixelFormat:int {
            rgba16float = 0,
            rgba32float,
            rgba8uint,
            rgba16uint,
            rgba32uint
        };

        enum class Type:int {
            i1d = 0,
            i2d,
            i3d
        };

        size_t width  = 0;
        size_t height = 0;
        size_t depth  = 1;
        PixelFormat pixel_format = PixelFormat::rgba32float;
        Type type = Type::i2d;
        MemFlags mem_flags = MemFlags::read_write;
    };

    struct TextureHolder;
    using Texture = std::shared_ptr<TextureHolder>;

    struct TextureHolder: public std::enable_shared_from_this<TextureHolder> {
    public:
        /***
         * Make a new empty read/write texture in command_queue
         * @param command_queue - gpu command_queue or context
         * @param desc - texture description
         * @param from_memory - from host memory that texture should be created
         * @return Texture object
         */
        static Texture Make(const void *command_queue, const TextureDesc &desc, float *from_memory = nullptr);

        Texture get_ptr() { return shared_from_this(); }

        virtual ~TextureHolder() = default;

        [[nodiscard]] virtual const void*  get_contents() const = 0;
        [[nodiscard]] virtual void*  get_contents() = 0;
        [[nodiscard]] virtual size_t get_width() const = 0;
        [[nodiscard]] virtual size_t get_height() const = 0;
        [[nodiscard]] virtual size_t get_depth() const = 0;
        [[nodiscard]] virtual size_t get_channels() const = 0;
        [[nodiscard]] virtual size_t get_length() const = 0;
        [[nodiscard]] virtual TextureDesc::PixelFormat get_pixel_format() const = 0;
        [[nodiscard]] virtual TextureDesc::Type get_type() const = 0;

        TextureHolder(const TextureHolder&) = delete;
        TextureHolder(TextureHolder&&) = delete;
        TextureHolder& operator=(const TextureHolder&) = delete;
        TextureHolder& operator=(TextureHolder&&) = delete;

    protected:
        TextureHolder() = default;
    };
}

