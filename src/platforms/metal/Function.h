//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/Log.h"
#include "dehancer/gpu/Paths.h"

#include "Command.h"

namespace dehancer::metal {
    
    typedef struct {
        unsigned long width;
        unsigned long height;
        unsigned long depth;
    } MTLSize;
    
    class Function {
    public:
    
        typedef std::function<dehancer::CommandEncoder::ComputeSize (dehancer::CommandEncoder& compute_encoder)> CommonEncodeHandler;
    
        struct ComputeSize {
            MTLSize threadsPerThreadgroup;
            MTLSize threadGroups;
        };

        struct PipelineState {
            void* pipeline;
            std::vector<dehancer::Function::ArgInfo> arg_list;
        };

        Function(dehancer::metal::Command* command, const std::string& kernel_name,  const std::string &library_path);
        void execute(const dehancer::Function::EncodeHandler& block);
       
        void execute(CommandEncoder::ComputeSize compute_size,
                     const dehancer::Function::VoidEncodeHandler& block);
        
        [[nodiscard]] const std::string& get_name() const;
        [[nodiscard]] std::vector<dehancer::Function::ArgInfo>& get_arg_info_list() const ;

        void set_current_pipeline() const ;

        [[nodiscard]] const std::string& get_library_path() const;
    
        [[nodiscard]] size_t get_block_max_size() const;
        [[nodiscard]] CommandEncoder::ComputeSize ask_compute_size(size_t width, size_t height, size_t depth) const;
        
        ~Function();

    private:
        dehancer::metal::Command* command_;
        std::string kernel_name_;
        std::string library_path_;

        typedef std::unordered_map<std::string, PipelineState> PipelineKernel;
        typedef std::unordered_map<void*, PipelineKernel> PipelineCache;

        mutable PipelineState pipelineState_;
        static PipelineCache pipelineCache_;
        static std::mutex mutex_;
    
        void execute_block(const CommonEncodeHandler& block);
    
    };
}


