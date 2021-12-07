//
// Created by denn on 30.12.2020.
//

#include "Texture.h"
#include "dehancer/gpu/Log.h"
#include <cuda_fp16.h>

namespace dehancer::cuda {
    
    
    TextureHolder::TextureHolder (const void *command_queue, const void *from_native_memory) :
            dehancer::TextureHolder(),
            Context(command_queue),
            desc_(),
            mem_(nullptr)
    {
      assert(mem_);
    }
    
    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc, const void *from_memory, bool is_device_buffer) :
            dehancer::TextureHolder(),
            Context(command_queue),
            desc_(desc),
            mem_(nullptr)
    {
      
      push();
      
      try {
  
        size_t pitch = 0;
        size_t dpitch = 0;
  
        switch (desc_.pixel_format) {
          
          case TextureDesc::PixelFormat::rgba32float:
            mem_ = make_texture<float4>();
            dpitch = pitch = sizeof(float4);
            break;
          
          case TextureDesc::PixelFormat::rgba16float:
            if (is_half_texture_allowed()) {
              mem_ = make_texture<float4, true>();
              pitch = sizeof(float4);
              dpitch = sizeof(float4) / 2;
            }
            else {
              mem_ = make_texture<float4, false>();
              dpitch = pitch = sizeof(float4);
            }
            break;
          
          case TextureDesc::PixelFormat::rgba32uint:
            mem_ = make_texture<uint4>();
            dpitch = pitch = sizeof(uint4);
            break;
          
          case TextureDesc::PixelFormat::rgba16uint:
            mem_ = make_texture<ushort4>();
            dpitch = pitch = sizeof(ushort4);
            break;
          
          case TextureDesc::PixelFormat::rgba8uint:
            mem_ = make_texture<uchar4>();
            dpitch = pitch = sizeof(uchar4);
            break;
        }
        
        if (from_memory) {
          if (desc_.type == TextureDesc::Type::i2d || desc_.type == TextureDesc::Type::i1d) {
            CHECK_CUDA(cudaMemcpy2DToArrayAsync(mem_->get_contents(),
                                                0, 0,
                                                from_memory,
                                                mem_->get_width() * pitch,
                                                mem_->get_width() * dpitch,
                                                mem_->get_height(),
                                                is_device_buffer ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                                                get_command_queue()));
          } else if (desc_.type == TextureDesc::Type::i3d) {
            
            
            cudaMemcpy3DParms cpy_params = {0};
            
            size_t nx = mem_->get_width(), ny = mem_->get_height(), nz = mem_->get_depth();
            
            cpy_params.srcPtr = make_cudaPitchedPtr((void *) from_memory, nx * pitch, nx * pitch, ny);
            cpy_params.dstArray = mem_->get_contents();
            
            cpy_params.extent = make_cudaExtent(nx, ny, nz);
            
            cpy_params.kind = is_device_buffer ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            
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
        auto mess = message_string(""
                                   "GPU out of memory. \r\n"
                                   "%s has total dedicated memory %i MB and %i MB is free. \r\n"
                                   "Please lower Project resolution, turn on Proxy Mode or upgrade your hardware",
                                   info.name, total, free_mem);
        dehancer::log::error(true, "CUDA make_texture error desc: %s", mess.c_str());
        throw dehancer::texture::memory_exception(mess);
      }
      pop();
    }
    
    TextureHolder::~TextureHolder() {
      push();
      mem_ = nullptr;
      pop();
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
  
      size_t dpitch = 0;
      size_t hpitch = sizeof(float4);
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          hpitch = dpitch = sizeof(float4);
          break;
        
        case TextureDesc::PixelFormat::rgba16float:
          dpitch = sizeof(float4);
          if (is_half_texture_allowed())
            dpitch/=2;
          break;
        
        case TextureDesc::PixelFormat::rgba32uint:
          hpitch = dpitch = sizeof(uint32_t[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba16uint:
          hpitch = dpitch = sizeof(uint16_t[4]);
          break;
        
        case TextureDesc::PixelFormat::rgba8uint:
          hpitch = dpitch = sizeof(uint8_t[4]);
          break;
      }
      
      try {
        push();
        CHECK_CUDA(cudaMemcpy2DFromArrayAsync(buffer,
                                              mem_->get_width() * hpitch,
                                              mem_->get_contents(),
                                              0, 0,
                                              mem_->get_width() * dpitch,
                                              mem_->get_height(),
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
          if (is_half_texture_allowed())
            return size * sizeof(float)/2;
          else
            return size * sizeof(float);
        
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
    
    dehancer::Error TextureHolder::copy_to_device (void *buffer) const {
      if (!buffer) {
        return Error(CommonError::OUT_OF_RANGE, "Target buffer undefined");
      }
  
      size_t dpitch = 0;
      size_t hpitch = sizeof(float4);
  
      switch (desc_.pixel_format) {
    
        case TextureDesc::PixelFormat::rgba32float:
          hpitch = dpitch = sizeof(float4);
          break;
    
        case TextureDesc::PixelFormat::rgba16float:
          dpitch = sizeof(float4);
          if (is_half_texture_allowed())
            dpitch/=2;
          break;
    
        case TextureDesc::PixelFormat::rgba32uint:
          hpitch = dpitch = sizeof(uint32_t[4]);
          break;
    
        case TextureDesc::PixelFormat::rgba16uint:
          hpitch = dpitch = sizeof(uint16_t[4]);
          break;
    
        case TextureDesc::PixelFormat::rgba8uint:
          hpitch = dpitch = sizeof(uint8_t[4]);
          break;
      }
  
      try {
        push();
        CHECK_CUDA(cudaMemcpy2DFromArrayAsync(buffer,
                                              mem_->get_width() * hpitch,
                                              mem_->get_contents(),
                                              0, 0,
                                              mem_->get_width() * dpitch,
                                              mem_->get_height(),
                                              cudaMemcpyDeviceToDevice,
                                              get_command_queue()));
        pop();
      }
      catch (const std::runtime_error &e) {
        return Error(CommonError::EXCEPTION, e.what());
      }
  
      return Error(CommonError::OK);
    }
  
  
}