//
// Created by denn nevera on 10/11/2020.
//

#include "Texture.h"
#include "dehancer/gpu/Log.h"
#include <cstring>

namespace dehancer::opencl {
    
    const void *TextureHolder::get_command_queue () const {
      return get_cl_command_queue();
    }
    
    TextureHolder::TextureHolder (const void *command_queue, const void *from_native_memory):
            dehancer::TextureHolder(),
            Context(command_queue),
            desc_(),
            memobj_(nullptr),
            releasable_(false)
    {
      assert(from_native_memory);
      memobj_ = static_cast<cl_mem>((void*)from_native_memory);
      
      cl_image_format image_format;
      last_error_ = clGetImageInfo(memobj_,CL_IMAGE_FORMAT, sizeof(image_format),&image_format,nullptr);
      if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to create texture from native memory: " + std::to_string(last_error_));
  
      if (CL_RGBA!=image_format.image_channel_order)
        if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to create texture from native memory: image channel order is not RGBA");
      
      size_t width=0, height=0, depth=0;
      last_error_ = clGetImageInfo(memobj_,CL_IMAGE_WIDTH, sizeof(width),&width,nullptr);
      if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to create texture from native memory: " + std::to_string(last_error_));
      
      last_error_ = clGetImageInfo(memobj_,CL_IMAGE_HEIGHT, sizeof(height),&height,nullptr);
      if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to create texture from native memory: " + std::to_string(last_error_));
      
      last_error_ = clGetImageInfo(memobj_,CL_IMAGE_DEPTH, sizeof(depth),&depth,nullptr);
      if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to create texture from native memory: " + std::to_string(last_error_));
  
      desc_.type = TextureDesc::Type::i1d;
      
      if (depth>1)
        desc_.type = TextureDesc::Type::i3d;
      else if (height>1)
        desc_.type = TextureDesc::Type::i2d;
      
      switch (image_format.image_channel_data_type) {
        case CL_FLOAT:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba32float;
        break;
    
        case CL_HALF_FLOAT:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba16float;
        break;
  
        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba32uint;
        break;
  
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba16uint;
        break;
    
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba8uint;
        break;
    
        default:
          throw std::runtime_error("Unsupported texture pixel format");
      }
  
      desc_.channels = 4;
      desc_.width = width;
      desc_.height = std::max((int)height,1);
      desc_.depth = std::max((int)depth,1);
    }
    
    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc, const void *from_memory, bool is_device_buffer) :
            dehancer::TextureHolder(),
            Context(command_queue),
            desc_(desc),
            memobj_(nullptr),
            releasable_(true)
    {
      cl_image_format format;
      cl_image_desc   image_desc;
      
      memset( &format, 0, sizeof( format ) );
      
      format.image_channel_order = CL_RGBA;
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          format.image_channel_data_type = CL_FLOAT;
          break;
        
        case TextureDesc::PixelFormat::rgba16float:
          format.image_channel_data_type = CL_HALF_FLOAT;
          break;
        
        case TextureDesc::PixelFormat::rgba32uint:
          format.image_channel_data_type = CL_UNSIGNED_INT32;
          break;
        
        case TextureDesc::PixelFormat::rgba16uint:
          format.image_channel_data_type = CL_UNSIGNED_INT16;
          break;
        
        case TextureDesc::PixelFormat::rgba8uint:
          format.image_channel_data_type = CL_UNSIGNED_INT8;
          break;
        
      }
      
      memset( &image_desc, 0, sizeof( image_desc ) );
      
      switch (desc_.type) {
        case TextureDesc::Type::i1d:
          image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
          break;
        case TextureDesc::Type::i2d:
          image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
          break;
        case TextureDesc::Type::i3d:
          image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
          break;
      }
      
      image_desc.image_width = desc_.width;
      image_desc.image_height = desc_.height;
      image_desc.image_depth = desc_.depth;
      
      cl_mem_flags mem_flags = 0;
      
      mem_flags |= TextureDesc::MemFlags::read_write & desc.mem_flags ? CL_MEM_READ_WRITE : 0;
      mem_flags |= TextureDesc::MemFlags::read_only & desc.mem_flags ? CL_MEM_READ_ONLY : 0;
      mem_flags |= TextureDesc::MemFlags::write_only & desc.mem_flags ? CL_MEM_WRITE_ONLY : 0;
      
      unsigned char* buffer = nullptr;
      
      if (from_memory) {
        buffer = reinterpret_cast<unsigned char *>((void *)from_memory);
        if (!is_device_buffer)
          mem_flags |= CL_MEM_COPY_HOST_PTR;
      }
  
      if (config::memory::alloc_host_ptr)
        mem_flags |= CL_MEM_ALLOC_HOST_PTR;
  
      if (is_device_buffer){
        buffer = nullptr;
      }
  
      last_error_ = CL_SUCCESS;
      
      memobj_ = clCreateImage(
              get_context(),
              mem_flags,
              &format,
              &image_desc,
              buffer,
              &last_error_);
  
      if (last_error_ != CL_SUCCESS)
        throw std::runtime_error("Unable to create texture: " + std::to_string(last_error_));
  
      if (is_device_buffer) {
        
        auto src = reinterpret_cast<cl_mem>((void*)from_memory);
        size_t dst_origin[3];
        size_t region[3];
        
        dst_origin[0] = 0; dst_origin[1] = 0; dst_origin[2] = 0;
        region[0] = get_width();
        region[1] = get_height();
        region[2] = get_depth();
        
        cl_mem_object_type m_type;
        last_error_ = clGetMemObjectInfo(src,CL_MEM_TYPE,sizeof(m_type),&m_type,nullptr);
        if (last_error_ != CL_SUCCESS)
          throw std::runtime_error("Unable to create texture: " + std::to_string(last_error_));
  
        if (CL_MEM_OBJECT_BUFFER==m_type) {
          last_error_ = clEnqueueCopyBufferToImage(
                  get_cl_command_queue() /* command_queue */,
                  src           /* src_buffer */,
                  memobj_       /* dst_image */,
                  0             /* src_offset */,
                  dst_origin    /* dst_origin[3] */,
                  region        /* region[3] */,
                  0             /* num_events_in_wait_list */,
                  nullptr       /* event_wait_list */,
                  nullptr       /* event */
          );
        }
        else if (CL_MEM_OBJECT_IMAGE2D==m_type){
          last_error_ = clEnqueueCopyImage(get_cl_command_queue()     /* command_queue */,
                                           src               /* src_image */,
                                           memobj_           /* dst_image */,
                                           dst_origin        /* src_origin[3] */,
                                           dst_origin        /* dst_origin[3] */,
                                           region            /* region[3] */,
                                           0                 /* num_events_in_wait_list */,
                                           nullptr           /* event_wait_list */,
                                           nullptr           /* event */);
  
        }
        else {
          throw std::runtime_error("Unable to copy image from object buffer type " + std::to_string(m_type));
        }
      }
      
      if (last_error_ != CL_SUCCESS)
        throw std::runtime_error("Unable to create texture: " + std::to_string(last_error_));
    }
    
    const void *TextureHolder::get_memory() const {
      return memobj_;
    }
    
    void *TextureHolder::get_memory() {
      return memobj_;
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
    
    TextureHolder::~TextureHolder() {
      if(releasable_ && memobj_)
        clReleaseMemObject(memobj_);
      #ifdef PRINT_KERNELS_DEBUG
      std::cout << "TextureHolder::~TextureHolder(" << memobj_ << ", "<< releasable_<<")"  << std::endl;
      #endif
    }
    
    dehancer::Error TextureHolder::get_contents(std::vector<float> &buffer) const {
      buffer.resize( get_length());
      return get_contents(buffer.data(), get_length());
    }
    
    dehancer::Error TextureHolder::get_contents(void *buffer, size_t length) const {
      size_t originst[3];
      size_t regionst[3];
      originst[0] = 0; originst[1] = 0; originst[2] = 0;
      regionst[0] = get_width();
      regionst[1] = get_height();
      regionst[2] = get_depth();
      size_t  rowPitch = 0;
      size_t  slicePitch = 0;
  
      if (length < this->get_length()) {
        return Error(CommonError::OUT_OF_RANGE, "Texture length greater then buffer length");
      }
      
      last_error_ = clEnqueueReadImage(
              get_cl_command_queue(),
              memobj_,
              CL_BLOCKING,
              originst,
              regionst,
              rowPitch,
              slicePitch,
              buffer,
              0,
              nullptr,
              nullptr );
      
      if (last_error_ != CL_SUCCESS) {
        return Error(CommonError::EXCEPTION, "Texture could not be read");
      }
      
      return Error(CommonError::OK);
    }
    
    dehancer::Error TextureHolder::copy_to_device (void *buffer) const {
      size_t originst[3];
      size_t regionst[3];
      originst[0] = 0; originst[1] = 0; originst[2] = 0;
      regionst[0] = get_width();
      regionst[1] = get_height();
      regionst[2] = get_depth();
      
      if (!buffer) {
        return Error(CommonError::OUT_OF_RANGE, "Target buffer is undefined");
      }
      
      auto dst = reinterpret_cast<cl_mem>(buffer);
      
      last_error_ =
              
              clEnqueueCopyImageToBuffer(
                      get_cl_command_queue() /* command_queue */,
                      memobj_             /* src_image */,
                      dst                 /* dst_buffer */,
                      originst            /* src_origin[3] */,
                      regionst            /* region[3] */,
                      0                   /* dst_offset */,
                      0                   /* num_events_in_wait_list */,
                      nullptr             /* event_wait_list */,
                      nullptr             /* event */);

      if (last_error_ != CL_SUCCESS) {
        return Error(CommonError::EXCEPTION, "Texture could not be copied to device");
      }
      
      return Error(CommonError::OK);
    }
  
}