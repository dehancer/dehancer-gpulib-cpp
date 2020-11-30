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

    class Function {
    public:

        struct ComputeSize {
            MTLSize threadsPerThreadgroup;
            MTLSize threadGroups;
        };

        struct PipelineState {
            id<MTLComputePipelineState> pipeline;
            std::vector<dehancer::Function::ArgInfo> arg_list;
        };

        Function(dehancer::metal::Command* command, const std::string& kernel_name,  const std::string &library_path);
        void execute(const dehancer::Function::FunctionHandler& block);

        [[nodiscard]] const std::string& get_name() const;
        [[nodiscard]] std::vector<dehancer::Function::ArgInfo>& get_arg_info_list() const ;

        void set_current_pipeline() const ;

        MTLSize get_threads_per_threadgroup(int w, int h, int d);
        MTLSize get_thread_groups(int w, int h, int d);
        ComputeSize get_compute_size(const CommandEncoder::Size size);

        ~Function();

    private:
        dehancer::metal::Command* command_;
        std::string kernel_name_;
        std::string library_path_;

        typedef std::unordered_map<std::string, PipelineState> PipelineKernel;
        typedef std::unordered_map<id<MTLCommandQueue>, PipelineKernel> PipelineCache;

        mutable PipelineState pipelineState_;
        static PipelineCache pipelineCache_;
        static std::mutex mutex_;
    };
}


