//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include <dehancer/gpu/Texture.h>
#include <dehancer/gpu/Function.h>
#include "Command.h"

namespace dehancer::cuda {

    class Function {
    public:
        Function(dehancer::cuda::Command* command,
                 const std::string& kernel_name,
                 const std::string &library_path
                 );
        void execute(const dehancer::Function::EncodeHandler& block);

        dehancer::cuda::Command* get_command() { return command_;}
        [[nodiscard]] const std::string& get_name() const;
        [[nodiscard]] const std::vector<dehancer::Function::ArgInfo>& get_arg_info_list() const ;

        ~Function();

    private:

        dehancer::cuda::Command* command_;
        std::string kernel_name_;
        std::string library_path_;
        CUfunction kernel_;
        mutable std::vector<dehancer::Function::ArgInfo> arg_list_;
        //CUcontext function_context_;
        size_t max_device_threads_;

        typedef std::unordered_map<std::string, CUfunction> KernelMap;
        typedef std::unordered_map<std::size_t, CUmodule> ProgamMap;

        static std::unordered_map<CUstream, KernelMap> kernel_map_;
        static std::unordered_map<CUstream, ProgamMap> module_map_;
        static std::mutex mutex_;
    };
}


