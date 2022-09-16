//
// Created by denn nevera on 10/11/2020.
//

#include "Context.h"

namespace dehancer::opencl {
    
    Context::Context(const void *command_queue): command_queue_(command_queue)
    {
      last_error_ = clGetCommandQueueInfo(get_command_queue(),
                                          CL_QUEUE_DEVICE,
                                          sizeof(cl_device_id),
                                          &device_id_,
                                          nullptr);
      
      if (last_error_ != CL_SUCCESS) {
        throw std::runtime_error("Unable to get OpenCL the device");
      }
      
      last_error_ = clGetCommandQueueInfo(get_command_queue(), CL_QUEUE_CONTEXT, sizeof(cl_context), &context_,
                                          nullptr);
      if (last_error_ != CL_SUCCESS) {
        throw std::runtime_error("Unable to get OpenCL context");
      }
    }
    
    cl_command_queue Context::get_command_queue() const {
      return static_cast<cl_command_queue>((void *) command_queue_);
    }
    
    cl_device_id Context::get_device_id() const {
      return device_id_;
    }
    
    cl_context Context::get_context() const {
      return context_;
    }
    
    dehancer::TextureInfo Context::get_texture_info (TextureDesc::Type texture_type) const {
      
      size_t width, height, depth;
     
      cl_device_info info_w = CL_DEVICE_IMAGE2D_MAX_WIDTH;
      cl_device_info info_h = CL_DEVICE_IMAGE2D_MAX_HEIGHT;
      cl_device_info info_d = CL_DEVICE_IMAGE2D_MAX_WIDTH;
    
      switch (texture_type) {
        case TextureDesc::Type::i1d:
        case TextureDesc::Type::i2d:
          break;
        case TextureDesc::Type::i3d:
          info_w = CL_DEVICE_IMAGE3D_MAX_WIDTH;
          info_h = CL_DEVICE_IMAGE3D_MAX_HEIGHT;
          info_d = CL_DEVICE_IMAGE3D_MAX_WIDTH;
          break;
      }
      last_error_ = clGetDeviceInfo(get_device_id(),info_w,sizeof(width),&width,nullptr);
      if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to get OpenCL context");

      last_error_ = clGetDeviceInfo(get_device_id(),info_h,sizeof(height),&height,nullptr);
      if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to get OpenCL context");

      last_error_ = clGetDeviceInfo(get_device_id(),info_d,sizeof(depth),&depth,nullptr);
      if (last_error_ != CL_SUCCESS) throw std::runtime_error("Unable to get OpenCL context");
      
      return {
        .max_width = width,
        .max_height = height,
        .max_depth = depth
      };
    }
}