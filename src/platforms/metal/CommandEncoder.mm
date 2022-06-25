//
// Created by denn nevera on 10/11/2020.
//

#include "CommandEncoder.h"
#import <Metal/Metal.h>

namespace dehancer::metal {
    void CommandEncoder::set(const Texture &texture, int index)  {
      if (command_encoder_) {
        auto ce = static_cast<id<MTLComputeCommandEncoder>>( (__bridge id) (void*)command_encoder_);
        auto t = static_cast<id <MTLTexture>>((__bridge id)texture->get_memory());
        [ce setTexture:t atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass texture to null kernel");
      }
    }

    void CommandEncoder::set(const void *bytes, size_t bytes_length, int index)  {
      if (command_encoder_){
        auto ce = static_cast<id<MTLComputeCommandEncoder>>( (__bridge id) (void*)command_encoder_);
        [ce setBytes:bytes length:bytes_length atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass bytes to null kernel");
      }
    }

    void CommandEncoder::set(const Memory &memory, int index) {
      if (command_encoder_) {
        auto ce = static_cast<id<MTLComputeCommandEncoder>>( (__bridge id) (void*)command_encoder_);
        auto buffer = static_cast<id <MTLBuffer>>((__bridge id)memory->get_memory());
        [ce setBuffer:buffer offset:0 atIndex:static_cast<NSUInteger>(index)];
      }
      else {
        throw std::runtime_error("Unable to pass buffer to null kernel");
      }
    }
    
    void CommandEncoder::set (const StreamSpace &p, int index) {
      StreamSpace copy = p;
      set(&copy, sizeof(copy), index);
    }
    
    size_t CommandEncoder::get_block_max_size () const {
      auto pipeline = reinterpret_cast<id<MTLComputePipelineState> >((__bridge id)pipeline_);
      return pipeline.maxTotalThreadsPerThreadgroup;
    }
    
    CommandEncoder::ComputeSize CommandEncoder::ask_compute_size (size_t width, size_t height, size_t depth) const {
      
      auto pipeline = reinterpret_cast<id<MTLComputePipelineState> >((__bridge id)pipeline_);
      auto workgroup_size = static_cast<size_t>(pipeline.maxTotalThreadsPerThreadgroup);
      auto execution_width = static_cast<size_t>(pipeline.threadExecutionWidth);
      
      ComputeSize compute_size {};
  
      size_t  gsize[2];
  
      if (workgroup_size <= 256)
      {
        gsize[0] = execution_width;
        gsize[1] = workgroup_size / execution_width;
      }
      else if (workgroup_size <= 1024)
      {
        gsize[0] = workgroup_size / execution_width;
        gsize[1] = execution_width;
      }
      else
      {
        gsize[0] = workgroup_size / 32;
        gsize[1] = 32;
      }
  
      compute_size.block.width  = gsize[0];
      compute_size.block.height = gsize[1];
  
      compute_size.grid.width = ((width + gsize[0] - 1) / gsize[0]);
      compute_size.grid.height = ((height + gsize[1] - 1) / gsize[1]);
  
      compute_size.threads_in_grid = compute_size.grid.width * compute_size.grid.height;
      
      compute_size.grid.depth = depth;
      compute_size.block.depth = 1;
  
      return compute_size;
    }
}