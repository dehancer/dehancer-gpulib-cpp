//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"

namespace dehancer::metal {

    std::mutex Function::mutex_;
    Function::PipelineCache Function::pipelineCache_ = Function::PipelineCache();
    inline static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, const std::string& kernel_name);

    void Function::execute(const dehancer::Function::FunctionHandler& block){

      auto pipeline = get_pipeline();

      if (!pipeline)
        throw std::runtime_error(error_string("Kernel %s could not execute with nil pipeline", kernel_name_.c_str()));

      id<MTLCommandQueue> queue = command_->get_command_queue();
      id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      [computeEncoder setComputePipelineState: pipeline];

      auto encoder = CommandEncoder(computeEncoder);

      auto from_block = block(encoder);

      if (!from_block)
        throw std::runtime_error(error_string("Kernel %s execute block error", kernel_name_.c_str()));

      auto texture = static_cast<id <MTLTexture>>((__bridge id)from_block->get_memory());

      auto grid = get_compute_size(texture);

      [computeEncoder dispatchThreadgroups:grid.threadGroups threadsPerThreadgroup: grid.threadsPerThreadgroup];
      [computeEncoder endEncoding];

      if (command_->get_wait_completed()) {
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder synchronizeTexture:texture slice:0 level:0];
        [blitEncoder endEncoding];
      }

      [commandBuffer commit];

      if (command_->get_wait_completed())
        [commandBuffer waitUntilCompleted];
    }

    Function::Function(dehancer::metal::Command *command, const std::string& kernel_name):
            command_(command),
            kernel_name_(kernel_name)
    {}

    id<MTLComputePipelineState> Function::get_pipeline() const {

      std::unique_lock<std::mutex> lock(Function::mutex_);

      id<MTLCommandQueue>            queue  = command_->get_command_queue();
      id<MTLDevice>                  device = command_->get_device();

      const auto it = Function::pipelineCache_.find(queue);

      if (it == Function::pipelineCache_.end())
      {
        if (!(pipelineState_  = make_pipeline(device, kernel_name_)))
          throw std::runtime_error(error_string("Make new pipeline for kernel %s error", kernel_name_.c_str()));
        Function::pipelineCache_[queue][kernel_name_] = pipelineState_;
      }
      else
      {

        const auto kernel_pit = it->second.find(kernel_name_);

        if (kernel_pit == it->second.end()) {
          if (!(pipelineState_  = make_pipeline(device, kernel_name_)))
          {
            throw std::runtime_error(error_string("Make new pipeline for kernel %s error", kernel_name_.c_str()));
          }
          Function::pipelineCache_[queue][kernel_name_] = pipelineState_;
        }
        else {
          pipelineState_ = kernel_pit->second;
        }
      }

      return pipelineState_;
    }

    MTLSize Function::get_threads_per_threadgroup(int w, int h, int d) {
      return MTLSizeMake(4, 4, d == 1 ? 1 : 4);
    }

    MTLSize Function::get_thread_groups(int w, int h, int d) {
      auto tpg = get_threads_per_threadgroup(w, h, d);
      return MTLSizeMake( (NSUInteger)(w/tpg.width), (NSUInteger)(h == 1 ? 1 : h/tpg.height), (NSUInteger)(d == 1 ? 1 : d/tpg.depth));
    }

    Function::ComputeSize Function::get_compute_size(const id<MTLTexture> &texture) {
      if ((int)texture.depth==1) {
        auto exeWidth = [pipelineState_ threadExecutionWidth];
        auto threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
        auto threadGroups     = MTLSizeMake((texture.width + exeWidth - 1)/exeWidth,
                                            texture.height, 1);
        return  {
                .threadsPerThreadgroup = threadGroupCount,
                .threadGroups = threadGroups
        };

      } else {
        auto threadsPerThreadgroup = get_threads_per_threadgroup((int)texture.width,
                                                                 (int)texture.height,
                                                                 (int)texture.depth) ;
        auto threadgroups  = get_thread_groups((int)texture.width,
                                               (int)texture.height,
                                               (int)texture.depth);
        return  {
                .threadsPerThreadgroup = threadsPerThreadgroup,
                .threadGroups = threadgroups
        };
      }
    }

    Function::~Function() {
    }

    NS_RETURNS_RETAINED inline static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, const std::string& kernel_name) {

      id<MTLComputePipelineState> pipelineState = nil;
      id<MTLLibrary>              metalLibrary;     // Metal library
      id<MTLFunction>             kernelFunction;   // Compute kernel

      NSError* err;

      std::string libpath = device::get_lib_path();

      if (libpath.empty()){
        if (!(metalLibrary    = [device newDefaultLibrary]))
          throw std::runtime_error(error_string("New default library cannot be created for kernel %s", kernel_name.c_str()));
      }
      else
      if (!(metalLibrary    = [device newLibraryWithFile:@(libpath.c_str()) error:&err]))
      {
        throw std::runtime_error(
                error_string("New library %s cannot be created for kernel %s: %s",
                             libpath.c_str(),
                             kernel_name.c_str(),
                             [[err localizedDescription] UTF8String]
                )
        );
      }


      if (!(kernelFunction  = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]]))
      {
        std::string next_kernel_space = "IMProcessing::";
        next_kernel_space.append(kernel_name);
        if (!(kernelFunction  = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:next_kernel_space.c_str()]])){
          [metalLibrary release];
          throw std::runtime_error(
                  error_string("New kernel %s cannot be created from library %s",
                               kernel_name.c_str(),
                               libpath.c_str()
                  )
          );
        }
      }


      if (!(pipelineState  = [device newComputePipelineStateWithFunction:kernelFunction error:&err]))
      {
        [metalLibrary release];
        [kernelFunction release];
        throw std::runtime_error(
                error_string("New pipeline cannot be created for %s from library %s: %s",
                             kernel_name.c_str(),
                             libpath.c_str(),
                             [[err localizedDescription] UTF8String]
                )
        );
      }

      //Release resources
      [metalLibrary release];
      [kernelFunction release];

      return pipelineState;
    }

}
