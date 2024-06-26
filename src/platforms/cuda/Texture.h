//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "Context.h"

#include "dehancer/gpu/kernels/cuda/texture1d.h"
#include "dehancer/gpu/kernels/cuda/texture2d.h"
#include "dehancer/gpu/kernels/cuda/texture3d.h"

namespace dehancer::cuda {

    struct TextureHolder: public dehancer::TextureHolder, public Context {
        
        TextureHolder(const void *command_queue, const TextureDesc &desc, const void *from_memory, bool is_device_buffer = false);
        TextureHolder(const void *command_queue, const void *from_native_memory);
        ~TextureHolder() override ;

        const void * get_command_queue() const override;
        
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
        TextureDesc desc_;
        dehancer::nvcc::texture* mem_;
        bool   releasable_ = true;

        template<class T, bool is_half = false>
        dehancer::nvcc::texture* make_texture() {
          switch (desc_.type) {
            case TextureDesc::Type::i1d:
              return new dehancer::nvcc::texture1d<T>(desc_.width);
            case TextureDesc::Type::i2d:
              return new dehancer::nvcc::texture2d<T,is_half>(desc_.width,desc_.height);
            case TextureDesc::Type::i3d:
              return new dehancer::nvcc::texture3d<T,is_half>(desc_.width,desc_.height,desc_.depth);
          }
      
          return new dehancer::nvcc::texture2d<T>(desc_.width,desc_.height);
        }
  
    };
}