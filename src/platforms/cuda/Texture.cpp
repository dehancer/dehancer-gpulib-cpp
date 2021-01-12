//
// Created by denn on 30.12.2020.
//

#include "Texture.h"
#include "dehancer/gpu/Log.h"
#include <cuda_fp16.h>

namespace dehancer::cuda {
    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc, const void *from_memory) :
            dehancer::TextureHolder(),
            Context(command_queue),
            desc_(desc),
            mem_(nullptr)
    {
      
      size_t pitch = 0;
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          mem_ = make_texture<float4>();
          pitch = sizeof(float4);
          break;
        
        case TextureDesc::PixelFormat::rgba16float:
          mem_ = make_texture<half[4]>();
          pitch = sizeof(half[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba32uint:
          mem_ = make_texture<uint32_t[4]>();
          pitch = sizeof(uint32_t[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba16uint:
          mem_ = make_texture<uint16_t[4]>();
          pitch = sizeof(uint16_t[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba8uint:
          mem_ = make_texture<uint8_t[4]>();
          pitch = sizeof(uint8_t[4]);
          break;
      }
      
      if (from_memory) {
        CHECK_CUDA(cudaMemcpy2DToArrayAsync(mem_->get_contents(),
                                            0, 0,
                                            from_memory,
                                            mem_->get_width() * pitch,
                                            mem_->get_width() * pitch, mem_->get_height(),
                                            cudaMemcpyHostToDevice,
                                            get_command_queue()));
      }
    }
    
    TextureHolder::~TextureHolder() {
      #ifdef PRINT_DEBUG
      dehancer::log::print("~TextureHolder[%s] desc: %ix%ix%i",  desc_.label.c_str(), desc_.width, desc_.height, desc_.depth);
      #endif
    };
    
    const void *TextureHolder::get_memory() const {
      return mem_.get();
    }
    
    void *TextureHolder::get_memory() {
      return mem_.get();
    }
    
    dehancer::Error TextureHolder::get_contents(std::vector<float> &buffer) const {
      buffer.resize( get_length());
      return get_contents(buffer.data(), get_length());
    }
    
    dehancer::Error TextureHolder::get_contents(void *buffer, size_t length) const {
      if (length < this->get_length()) {
        return Error(CommonError::OUT_OF_RANGE, "Texture length greater then buffer length");
      }
      
      size_t pitch = 0;
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          pitch = sizeof(float4);
          break;
        
        case TextureDesc::PixelFormat::rgba16float:
          pitch = sizeof(half[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba32uint:
          pitch = sizeof(uint32_t[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba16uint:
          pitch = sizeof(uint16_t[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba8uint:
          pitch = sizeof(uint8_t[4]);
          break;
      }
      
      try {
        CHECK_CUDA(cudaMemcpy2DFromArrayAsync(buffer,
                                              mem_->get_width() * pitch,
                                              mem_->get_contents(),
                                              0, 0, mem_->get_width() * pitch, mem_->get_height(),
                                              cudaMemcpyDeviceToHost,
                                              get_command_queue()));
      }
      catch (const std::runtime_error &e) {
        return Error(CommonError::EXCEPTION, e.what());
      }
      
      return Error(CommonError::OK);
    }
    
    size_t TextureHolder::get_width() const {
      return desc_.width;
    }
    
    size_t TextureHolder::get_height() const {
      return desc_.height;
    }
    
    size_t TextureHolder::get_depth() const {
      return desc_.depth;
    }
    
    size_t TextureHolder::get_channels() const {
      return desc_.channels;
    }
    
    size_t TextureHolder::get_length() const {
      
      size_t size = desc_.width*desc_.depth*desc_.height*get_channels();
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          return size * sizeof(float);
        
        case TextureDesc::PixelFormat::rgba16float:
          return size * sizeof(float)/2;
        
        case TextureDesc::PixelFormat::rgba32uint:
          return size * sizeof(uint32_t);
        
        case TextureDesc::PixelFormat::rgba16uint:
          return size * sizeof(uint16_t);
        
        case TextureDesc::PixelFormat::rgba8uint:
          return size * sizeof(uint8_t);
      }
    }
    
    TextureDesc::PixelFormat TextureHolder::get_pixel_format() const {
      return desc_.pixel_format;
    }
    
    TextureDesc::Type TextureHolder::get_type() const {
      return desc_.type;
    }
}