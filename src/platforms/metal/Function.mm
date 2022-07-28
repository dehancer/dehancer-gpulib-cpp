//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#import <Metal/Metal.h>

namespace dehancer::metal {
    
    std::mutex Function::mutex_;
    Function::PipelineCache Function::pipelineCache_ = Function::PipelineCache();
    
    inline static Function::PipelineState make_pipeline(
            id<MTLDevice> device,
            const std::string& kernel_name,
            const std::string& library_path
    );
    
    const std::string &Function::get_library_path () const {
      return library_path_;
    }
    
    void Function::execute_block (const Function::CommonEncodeHandler &block) {
      
      auto queue = static_cast<id<MTLCommandQueue>>( (__bridge id) command_->get_command_queue());
  
      id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
  
      auto c_pipeline = reinterpret_cast<id<MTLComputePipelineState> >((__bridge id)pipelineState_.pipeline);
      [computeEncoder setComputePipelineState: c_pipeline];
  
      auto encoder = CommandEncoder(command_, pipelineState_.pipeline, computeEncoder);
      dehancer::CommandEncoder* encoder_ref = &encoder;
      
      auto compute_size = block(encoder);
  
      #ifdef PRINT_KERNELS_DEBUG
      size_t buffer_size = compute_size.threads_in_grid*257*4*sizeof(unsigned int);
      std::cout << " #Function " << kernel_name_
                << " global: "
                << compute_size.grid.width << "x" << compute_size.grid.height << "x" << compute_size.grid.depth
                << "  local: "
                << compute_size.block.width << "x" << compute_size.block.height << "x" << compute_size.block.depth
                << "  num_groups: "
                << compute_size.threads_in_grid
                << "  buffer size: "
                <<     buffer_size << "b" << ", " << buffer_size/1024/1204 << "Mb"
                << std::endl;
      #endif
  
      ComputeSize grid =
              {
                      .threadsPerThreadgroup = {
                              .width = compute_size.block.width,
                              .height = compute_size.block.height,
                              .depth = compute_size.block.depth
                      },
                      .threadGroups = {
                              .width = compute_size.grid.width,
                              .height = compute_size.grid.height,
                              .depth = compute_size.grid.depth
                      }
              };
  
      [computeEncoder
              dispatchThreadgroups: {grid.threadGroups.width, grid.threadGroups.height, grid.threadGroups.depth}
             threadsPerThreadgroup: {grid.threadsPerThreadgroup.width, grid.threadsPerThreadgroup.height, grid.threadsPerThreadgroup.depth}];
      [computeEncoder endEncoding];
  
      [commandBuffer commit];
  
      if (command_->get_wait_completed())
        [commandBuffer waitUntilCompleted];
    }
    
    void Function::execute (CommandEncoder::ComputeSize compute_size,
                            const dehancer::Function::VoidEncodeHandler &block)
    {
      if (!block) return;
      
      execute_block([this,block,compute_size](dehancer::CommandEncoder& encoder){
          block(encoder);
          return compute_size;
      });
    }
    
    void Function::execute(const dehancer::Function::EncodeHandler& block){
      execute_block([this,block](dehancer::CommandEncoder& encoder){
          auto from_block = block(encoder);
          return encoder.ask_compute_size(from_block);
      });
    }
    
    Function::Function(dehancer::metal::Command *command, const std::string& kernel_name,  const std::string &library_path):
            command_(command),
            kernel_name_(kernel_name),
            library_path_(library_path),
            pipelineState_{nullptr, {}}
    {
      set_current_pipeline();
      if (!pipelineState_.pipeline)
        throw std::runtime_error(error_string("Kernel %s could not execute with nil pipeline", kernel_name_.c_str()));
    }
    
    void Function::set_current_pipeline() const {
      
      std::unique_lock<std::mutex> lock(Function::mutex_);
  
      auto queue = static_cast<id<MTLCommandQueue>>( (__bridge id) command_->get_command_queue());
      auto device = static_cast<id<MTLDevice>>( (__bridge id) command_->get_device());
      
      const auto it = Function::pipelineCache_.find(queue);
      
      if (it == Function::pipelineCache_.end())
      {
        pipelineState_  = make_pipeline(device, kernel_name_, library_path_);
        if (!pipelineState_.pipeline)
          throw std::runtime_error(error_string("Make new pipeline for kernel %s error", kernel_name_.c_str()));
        Function::pipelineCache_[queue][kernel_name_] = pipelineState_;
      }
      else
      {
        
        const auto kernel_pit = it->second.find(kernel_name_);
        
        if (kernel_pit == it->second.end()) {
          pipelineState_  = make_pipeline(device, kernel_name_, library_path_);
          if (!pipelineState_.pipeline)
          {
            throw std::runtime_error(error_string("Make new pipeline for kernel %s error", kernel_name_.c_str()));
          }
          Function::pipelineCache_[queue][kernel_name_] = pipelineState_;
        }
        else {
          pipelineState_ = kernel_pit->second;
        }
      }
    }
    
    const std::string &Function::get_name() const {
      return kernel_name_;
    }
    
    std::vector<dehancer::Function::ArgInfo>& Function::get_arg_info_list() const {
      return pipelineState_.arg_list;
    }
    
    size_t Function::get_block_max_size () const {
      auto encoder = CommandEncoder(command_, pipelineState_.pipeline, nullptr);
      return encoder.get_block_max_size();
    }

    CommandEncoder::ComputeSize Function::ask_compute_size (size_t width, size_t height, size_t depth) const {
      auto encoder = CommandEncoder(command_, pipelineState_.pipeline, nullptr);
      return encoder.ask_compute_size (width, height, depth);
    }
    
    
    Function::~Function() = default;
    
    inline static Function::PipelineState make_pipeline(
            id<MTLDevice> device,
            const std::string& kernel_name,
            const std::string& library_path
    ) {
      
      id<MTLComputePipelineState> pipelineState = nil;
      id<MTLLibrary>              metalLibrary;     // Metal library
      id<MTLFunction>             kernelFunction;   // Compute kernel
      
      NSError* err;
      
      std::string libpath = library_path.empty() ? device::get_lib_path() : library_path;
      
      Function::PipelineState state{nullptr, {}};
      
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
      
      
      MTLAutoreleasedComputePipelineReflection reflection;
      if (!(pipelineState  = [device newComputePipelineStateWithFunction:kernelFunction
                                                                 options:MTLPipelineOptionArgumentInfo
                                                              reflection:&reflection
                                                                   error:&err]))
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
      
      state.pipeline = pipelineState;
      for(MTLArgument* a in reflection.arguments) {
        dehancer::Function::ArgInfo info {
                .name = [a.name UTF8String],
                .index = static_cast<uint>(a.index),
                .type_name = std::to_string([a type])
        };
        state.arg_list.push_back(info);
      }
      
      //Release resources
      [metalLibrary release];
      [kernelFunction release];
      
      return state;
    }
  
}
