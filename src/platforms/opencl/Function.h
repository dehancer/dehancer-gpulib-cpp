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
        Function(dehancer::opencl::Command* command,
                 const std::string& kernel_name,
                 const std::string &library_path
                 );
        void execute(const dehancer::Function::EncodeHandler& block);

        [[nodiscard]] const std::string& get_name() const;
        [[nodiscard]] const std::vector<dehancer::Function::ArgInfo>& get_arg_info_list() const ;

        ~Function();

    private:

        dehancer::opencl::Command* command_;
        std::string kernel_name_;
        std::string library_path_;
        cl_kernel kernel_;
        std::shared_ptr<CommandEncoder> encoder_;
        mutable std::vector<dehancer::Function::ArgInfo> arg_list_;

        typedef std::unordered_map<std::string, cl_kernel> KernelMap;
        typedef std::unordered_map<std::size_t, cl_program> ProgamMap;

        static std::unordered_map<cl_command_queue, KernelMap> kernel_map_;
        static std::unordered_map<cl_command_queue, ProgamMap> program_map_;
        static std::mutex mutex_;

    };
}


