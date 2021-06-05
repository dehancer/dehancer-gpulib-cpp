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
      
      push();
      
      try {
        
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
          if (desc_.type == TextureDesc::Type::i2d || desc_.type == TextureDesc::Type::i1d) {
            CHECK_CUDA(cudaMemcpy2DToArrayAsync(mem_->get_contents(),
                                                0, 0,
                                                from_memory,
                                                mem_->get_width() * pitch,
                                                mem_->get_width() * pitch,
                                                mem_->get_height(),
                                                cudaMemcpyHostToDevice,
                                                get_command_queue()));
          } else if (desc_.type == TextureDesc::Type::i3d) {
            
            
            cudaMemcpy3DParms cpy_params = {0};
            
            size_t nx = mem_->get_width(), ny = mem_->get_height(), nz = mem_->get_depth();
            
            cpy_params.srcPtr = make_cudaPitchedPtr((void *) from_memory, nx * pitch, nx * pitch, ny);
            cpy_params.dstArray = mem_->get_contents();
            
            cpy_params.extent = make_cudaExtent(nx, ny, nz);
            
            cpy_params.kind = cudaMemcpyHostToDevice;
            
            cudaMemcpy3DAsync(&cpy_params, get_command_queue());
          }
        }
      }
      catch (const std::runtime_error& e) {
        dehancer::log::error(true, "CUDA make_texture error: %s", e.what());
        size_t total=0, free_mem=0;
        get_mem_info(total,free_mem);
        cudaDeviceProp info{};
        get_device_info(info);
        total /= 1024*1024;
        free_mem /= 1024*1024;
        auto mess = error_string(""
                                 "\nGPU out of memory"
                                 "\n%s has total dedicated memory %i MB and %i MB is free\n"
                                 "\n"
                                 "Please lower project resolution, turn on Proxy Mode or upgrade your hardware",
                                 info.name, total, free_mem);
        dehancer::log::error(true, "CUDA make_texture error desc: %s", mess.c_str());
        throw dehancer::texture::memory_exception(mess);
      }
      pop();
    }
    
    TextureHolder::~TextureHolder() = default;
    
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
        push();
        CHECK_CUDA(cudaMemcpy2DFromArrayAsync(buffer,
                                              mem_->get_width() * pitch,
                                              mem_->get_contents(),
                                              0, 0, mem_->get_width() * pitch, mem_->get_height(),
                                              cudaMemcpyDeviceToHost,
                                              get_command_queue()));
        pop();
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