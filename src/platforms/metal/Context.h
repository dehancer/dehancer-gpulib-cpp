//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include <cstdlib>
#include <dehancer/gpu/Texture.h>

namespace dehancer::metal {
    
    class Context {

    public:
        explicit Context(const void *command_queue);
        [[nodiscard]] void* get_command_queue() const;
        [[nodiscard]] void* get_device() const;
        [[nodiscard]] bool has_unified_memory() const;
        [[nodiscard]] TextureInfo get_texture_info(TextureDesc::Type texture_type) const;
        
    private:
        const void* command_queue_;
    };
}

