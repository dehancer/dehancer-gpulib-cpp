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

        struct ComputeSize {
            MTLSize threadsPerThreadgroup;
            MTLSize threadGroups;
        };

    public:
        Function(dehancer::metal::Command* command, const std::string& kernel_name);
        void execute(const dehancer::Function::FunctionHandler& block);

        id<MTLComputePipelineState> get_pipeline() const ;

        MTLSize get_threads_per_threadgroup(int w, int h, int d);
        MTLSize get_thread_groups(int w, int h, int d);
        ComputeSize get_compute_size(const id<MTLTexture> &texture);

        ~Function();

    private:
        dehancer::metal::Command* command_;
        std::string kernel_name_;

        typedef std::unordered_map<std::string, id<MTLComputePipelineState>> PipelineKernel;
        typedef std::unordered_map<id<MTLCommandQueue>, PipelineKernel> PipelineCache;

        mutable id<MTLComputePipelineState> pipelineState_;
        static PipelineCache pipelineCache_;
        static std::mutex mutex_;
    };
}


