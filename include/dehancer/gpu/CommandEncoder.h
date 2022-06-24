//
// Created by denn nevera on 20/11/2020.
//

#pragma once

#include "dehancer/gpu/Typedefs.h"
#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Memory.h"
#include "StreamSpace.h"

namespace dehancer {
    /***
    * Command encoder offers methods to bind parameters that are objects
    * of host system and placed in CPU memory and GPU kernel functions.
    */
    class CommandEncoder {
    public:
    
        struct Size {
            size_t width;
            size_t height;
            size_t depth;

            static inline Size From(const Texture& t) {
              if (!t) return  {0,0,0};
              return {t->get_width(), t->get_height(), t->get_depth()};
            };
    
            static inline Size From(int w, int h, int d=1) {
              return {static_cast<size_t>(w), static_cast<size_t>(h), static_cast<size_t>(d)};
            };
    
            static inline Size From(size_t w, size_t h, size_t d=1) {
              return {static_cast<size_t>(w), static_cast<size_t>(h), static_cast<size_t>(d)};
            };
        };
    
        struct ComputeSize {
            Size   grid;
            Size   block;
            size_t threads_in_grid;
        };
        
        /**
         * Get the current device max number of threads in a block
         *
         * @return max threads number
         */
        [[nodiscard]] virtual size_t get_block_max_size() const = 0;
        
        /***
         * Ask to calculate the best solution for computation grid size
         * @param width source data width, i.e. the texture width or xD memory width
         * @param height source data width
         * @param depth source data depth
         * @return computation size
         */
        [[nodiscard]] virtual ComputeSize ask_compute_size(size_t width, size_t height, size_t depth) const = 0;
        
        /***
         * Ask to calculate the best solution for computation grid size for a texture defined by texture size
         * @param texture_size texture size
         * @return computation size
         */
        virtual ComputeSize ask_compute_size(Size texture_size);
        
        /***
         * Ask to calculate the best solution for computation grid size for a texture
         * @param source texture source
         * @return computation size
         */
        virtual ComputeSize ask_compute_size(const Texture& source);
    
        
        /***
         * Bind texture object with kernel argument placed at defined index. @see Texture
         * @param texture - texture object
         * @param index - index place at kernel parameter list
         */
        virtual void set(const Texture& texture, int index) = 0;

        /***
        * Bind memory object with kernel argument placed at defined index. @see Memory
        * @param texture - texture object
        * @param index - index place at kernel parameter list
        */
        virtual void set(const Memory& memory, int index) = 0;

        /***
         * Bind raw bytes with kernel argument placed at defined index. @see Texture
         * @param bytes - host memory buffer
         * @param bytes_length - buffer length
         * @param index - index place at kernel parameter list
         */
        virtual void set(const void* bytes, size_t bytes_length, int index) = 0;

        virtual void set(bool p, int index);
        virtual void set(char p, int index);
        virtual void set(int8_t p, int index);
        virtual void set(int16_t p, int index);
        virtual void set(int32_t p, int index);
        virtual void set(uint8_t p, int index);
        virtual void set(uint16_t p, int index);
        virtual void set(uint32_t p, int index);
        
        virtual void set(float p, int index);
        
        virtual void set(const float2& p, int index);
        virtual void set(const float3& p, int index);
        virtual void set(const float4& p, int index);
    
        virtual void set(const float2x2& m, int index);
        virtual void set(const float3x3& m, int index);
        virtual void set(const float4x4& m, int index);
    
        virtual void set(const math::uint2& p, int index);
        virtual void set(const math::uint3& p, int index);
        virtual void set(const math::uint4& p, int index);
    
        virtual void set(const math::int2& p, int index);
        virtual void set(const math::int3& p, int index);
        virtual void set(const math::int4& p, int index);
    
        virtual void set(const math::bool2& p, int index);
        virtual void set(const math::bool3& p, int index);
        virtual void set(const math::bool4& p, int index);
        
        virtual void set(const dehancer::StreamSpace& p, int index);
    
        virtual ~CommandEncoder() = default;
    };
}