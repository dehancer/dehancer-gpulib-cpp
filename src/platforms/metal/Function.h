//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include <dehancer/gpu/Texture.h>
#include <dehancer/gpu/Function.h>
#include "Command.h"

namespace dehancer::metal {

    class Function {
    public:
        Function(dehancer::metal::Command* command, const std::string& kernel_name);
        void execute(const dehancer::Function::FunctionHandler& block);

        ~Function();

    private:
        dehancer::metal::Command* command_;
        std::string kernel_name_;
        std::shared_ptr<CommandEncoder> encoder_;
    };
}


