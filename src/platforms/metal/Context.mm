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

    //id<MTLCommandQueue> Context::get_command_queue() const {
    void* Context::get_command_queue() const {
      return static_cast<id<MTLCommandQueue>>( (__bridge id) command_queue_);
    }

    //id<MTLDevice> Context::get_device() const {
    void* Context::get_device() const {
      return //get_command_queue().device;
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
}