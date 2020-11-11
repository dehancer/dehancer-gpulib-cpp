//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include <dehancer/gpu/Texture.h>
#include <dehancer/gpu/Function.h>
#include "OCLCommand.h"

namespace dehancer::opencl {

    class Function {
    public:
        Function(dehancer::opencl::Command* command, const std::string& kernel_name);
        void execute(const dehancer::Function::FunctionHandler& block);

    private:
        dehancer::opencl::Command* command_;
        std::string kernel_name_;
        cl_program program_;
        cl_kernel kernel_;
        std::shared_ptr<CommandEncoder> encoder_;
    };
}


