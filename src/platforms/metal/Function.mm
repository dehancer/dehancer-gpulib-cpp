//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"

namespace dehancer::metal {

    void Function::execute(const dehancer::Function::FunctionHandler &block) {

      auto texture = block(*encoder_);

      if (!texture) return;
    }

    Function::Function(dehancer::metal::Command *command, const std::string& kernel_name):
            command_(command),
            kernel_name_(kernel_name),
            encoder_(nullptr)
    {
    }

    Function::~Function() {
    }
}
