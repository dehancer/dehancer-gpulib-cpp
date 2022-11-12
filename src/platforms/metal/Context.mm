//
// Created by denn nevera on 10/11/2020.
//

#include "Context.h"
#import <Metal/Metal.h>

namespace dehancer::metal {
    
    Context::Context(const void *command_queue):
            command_queue_(static_cast<id<MTLCommandQueue>>( (__bridge id) (void*)command_queue))
    {
    }
    
    void* Context::get_command_queue() const {
      return static_cast<id<MTLCommandQueue>>( (__bridge id) command_queue_);
    }
    
    void* Context::get_device() const {
      return
              static_cast<id<MTLCommandQueue>>( (__bridge id) command_queue_).device;
    }
    
    bool Context::has_unified_memory () const {
      auto* device = static_cast<id<MTLCommandQueue>>( (__bridge id) command_queue_).device;
      if([device respondsToSelector:@selector(hasUnifiedMemory)]) {
        if (@available(macOS 10.15, iOS 13.0, *)) {
          return [device hasUnifiedMemory];
        } else {
          return false;
        }
      }
      return false;
    }
    
    TextureInfo Context::get_texture_info (TextureDesc::Type texture_type) const {
      
      size_t size;
      
      auto device = static_cast<id <MTLCommandQueue>>((__bridge id) command_queue_).device;
      
      if (![[device class] instancesRespondToSelector:@selector(supportsFamily:)]) {
        size = 16384;
      }
//      if (@available(iOS 13, macOS 10.15, *)) {
//      #if DEHANCER_ARCH_IS_X86_64 and DEHANCER_ARCH_MACOS
//      size = 16384;
//      #else
      else if (texture_type == TextureDesc::Type::i2d) {
        if (
                [device supportsFamily:MTLGPUFamilyApple3]
                ||
                [device supportsFamily:MTLGPUFamilyApple4]
                ||
                [device supportsFamily:MTLGPUFamilyApple5]
                #if !DEHANCER_ARCH_IS_X86_64
                ||
                  [device supportsFamily:MTLGPUFamilyApple6]
                  ||
                  [device supportsFamily:MTLGPUFamilyApple7]
                #endif
                ||
                [device supportsFamily:MTLGPUFamilyMac2]
                ||
                [device supportsFamily:MTLGPUFamilyMacCatalyst2]
                ) {
          size = 16384;
        } else
          size = 8192;
      } else if (texture_type == TextureDesc::Type::i1d) {
        if (
                [device supportsFamily:MTLGPUFamilyApple3]
                ||
                [device supportsFamily:MTLGPUFamilyApple4]
                ||
                [device supportsFamily:MTLGPUFamilyApple5]
                #if !DEHANCER_ARCH_IS_X86_64
                ||
                  [device supportsFamily:MTLGPUFamilyApple6]
                  ||
                  [device supportsFamily:MTLGPUFamilyApple7]
                #endif
                ||
                [device supportsFamily:MTLGPUFamilyMac2]
                ||
                [device supportsFamily:MTLGPUFamilyMacCatalyst2]
                ) {
          size = 16384;
        } else
          size = 8192;
      } else {
        size = 2048;
      }
//      #endif
      
      return {
              .max_width = size,
              .max_height = size,
              .max_depth = size
      };
    }
}