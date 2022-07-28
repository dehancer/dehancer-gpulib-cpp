//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "Context.h"

namespace dehancer::metal {
    
    struct TextureItem {
        size_t         hash = 0;
        void*          texture = nullptr;
        bool           releasable = true;
        ~TextureItem();
    };
    
    struct TextureHolder: public dehancer::TextureHolder, public Context {
        TextureHolder(const void *command_queue, const TextureDesc &desc, const void *from_memory, bool is_device_buffer);
        TextureHolder(const void *command_queue, const void *from_memory);
        ~TextureHolder() override ;

        [[nodiscard]] const void*  get_memory() const override;
        [[nodiscard]] void*  get_memory() override;
        dehancer::Error get_contents(std::vector<float>& buffer) const override;
        dehancer::Error get_contents(void* buffer, size_t length) const override;
        [[nodiscard]] size_t get_width() const override;
        [[nodiscard]] size_t get_height() const override;
        [[nodiscard]] size_t get_depth() const override;
        [[nodiscard]] size_t get_channels() const override;
        [[nodiscard]] size_t get_length() const override;
        [[nodiscard]] TextureDesc::PixelFormat get_pixel_format() const override;
        [[nodiscard]] TextureDesc::Type get_type() const override;

        dehancer::Error copy_to_device(void* buffer) const override;
    
        TextureDesc get_desc() const override { return desc_;}
        
    private:
        TextureDesc  desc_;
        std::shared_ptr<TextureItem> texture_item_;
    };
}