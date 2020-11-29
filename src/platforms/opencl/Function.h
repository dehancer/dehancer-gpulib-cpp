//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include <dehancer/gpu/Texture.h>
#include <dehancer/gpu/Function.h>
#include "Command.h"

namespace dehancer::opencl {

    class Function {
    public:
        Function(dehancer::opencl::Command* command, const std::string& kernel_name);
        void execute(const dehancer::Function::FunctionHandler& block);

        [[nodiscard]] const std::string& get_name() const;
        [[nodiscard]] const std::vector<dehancer::Function::ArgInfo>& get_arg_info_list() const ;

        ~Function();

    private:

        //typedef std::unordered_map<id<MTLCommandQueue>, PipelineKernel> PipelineCache;


        dehancer::opencl::Command* command_;
        std::string kernel_name_;
        //cl_program program_;
        cl_kernel kernel_;
        std::shared_ptr<CommandEncoder> encoder_;
        mutable std::vector<dehancer::Function::ArgInfo> arg_list_;

        typedef std::unordered_map<std::string, cl_kernel> KernelMap;

        static std::unordered_map<cl_command_queue, cl_device_id> device_id_map_;
        static std::unordered_map<cl_command_queue, KernelMap> kernel_map_;
        static std::mutex mutex_;

    };
}


