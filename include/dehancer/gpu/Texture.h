//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include <memory>
#include <utility>
#include "dehancer/Common.h"
#include "dehancer/gpu/Memory.h"

namespace dehancer {
    
    struct TextureHolder;
    
    /***
     * Texture pointer object
     */
    using Texture = std::shared_ptr<TextureHolder>;
    
    namespace texture {
        class memory_exception: public std::exception {
        public:
            explicit memory_exception(std::string  message): message_(std::move(message)){}
            const char * what() const noexcept override { return message_.c_str(); };

        private:
            std::string message_;
        };
    }
    
    /***
     * Texture description
     */
    struct TextureDesc {
        
        /***
         * Texture memory flags type
         */
        enum MemFlags:uint32_t {
            read_write = (1 << 0),
            write_only = (1 << 1),
            read_only  = (1 << 2),
        };
        
        /***
         * Texture pixel packing format type. By default rgba32float.
         */
        enum class PixelFormat:int {
            rgba16float = 0,
            rgba32float,
            rgba8uint,
            rgba16uint,
            rgba32uint
        };
        
        /***
         * Texture dimension type. By default 2D
         */
        enum class Type:int {
            i1d = 0,
            i2d,
            i3d
        };
        
        /***
         * Texture width
         */
        size_t width  = 0;
        
        /***
         * Texture height
         */
        size_t height = 0;
        
        /***
         * Texture depth. By default texture is 2D and depth is 1
         */
        size_t depth  = 1;
        
        /***
         * Current version supports only RGBA 4 channels textures
         */
        size_t channels = 4;
        
        /***
         * Pixel format value.
         */
        PixelFormat pixel_format = PixelFormat::rgba32float;
        
        /***
         * Texture dimension type value
         */
        Type type = Type::i2d;
        
        /***
         * Texture memory flags options
         */
        MemFlags mem_flags = MemFlags::read_write;
        
        /***
         * Debug info
         */
        std::string label;
        
        [[nodiscard]] size_t get_hash() const ;
        
        Texture make(const void *command_queue, const float *from_memory = nullptr) const;
    };
    
    bool operator==(const TextureDesc& lhs, const TextureDesc& rhs);
    bool operator!=(const TextureDesc& lhs, const TextureDesc& rhs);
    
    /***
     * Texture object holder. U must use only Texture pointer object.
     */
    struct TextureHolder: public std::enable_shared_from_this<TextureHolder> {
    public:
        /***
         * Make a new empty read/write texture in command_queue
         * @param command_queue - device command_queue or context
         * @param desc - texture description
         * @param from_memory - from host memory that texture should be created
         * @return Texture object
         */
        static Texture Make(const void *command_queue, const TextureDesc &desc, const float *from_memory = nullptr);
        
        /***
         * Get a weak shared pointer to texture object.
         * @return
         */
        Texture get_ptr() { return shared_from_this(); }
        
        virtual ~TextureHolder();
        
        /***
         * Get platform specific handler of texture placed in device memory.
         * @return device memory handler
         */
        [[nodiscard]] virtual const void*  get_memory() const = 0;
        /***
        * Get platform specific handler of texture placed in device memory.
        * @return device memory handler
        */
        [[nodiscard]] virtual void*  get_memory() = 0;
        
        /***
         * Copy contents of texture object to host memory buffer as as a sequential array of floats.
         * @param buffer
         * @return expected Error object descriptor or Error::OK
         */
        virtual Error get_contents(std::vector<float>& buffer) const = 0;
        
        virtual Error get_contents(void* buffer, size_t length) const = 0;
        
        /***
         * Get texture width.
         * @return width
         */
        [[nodiscard]] virtual size_t get_width() const = 0;
        
        /***
         * Get texture height.
         * @return height
         */
        [[nodiscard]] virtual size_t get_height() const = 0;
        
        /***
        * Get texture depth.
        * @return depth
        */
        [[nodiscard]] virtual size_t get_depth() const = 0;
        
        /***
        * Get texture number of channels.
        * @return channels
        */
        [[nodiscard]] virtual size_t get_channels() const = 0;
        
        /***
         * Get texture size in bytes.
         * @return number of bytes
         */
        [[nodiscard]] virtual size_t get_length() const = 0;
        
        /***
         * Get texture pixel format.
         * @return
         */
        [[nodiscard]] virtual TextureDesc::PixelFormat get_pixel_format() const = 0;
        
        /***
         * Get texture dimension type.
         * @return
         */
        [[nodiscard]] virtual TextureDesc::Type get_type() const = 0;
        
        virtual TextureDesc get_desc() const = 0;
        
        TextureHolder(const TextureHolder&) = delete;
        TextureHolder(TextureHolder&&) = delete;
        TextureHolder& operator=(const TextureHolder&) = delete;
        TextureHolder& operator=(TextureHolder&&) = delete;
    
    protected:
        TextureHolder() = default;
    };
}

