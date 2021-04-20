//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "utils.h"

namespace dehancer::cuda {

    class Context {

    public:
        explicit Context(const void *command_queue);
        [[nodiscard]] CUstream get_command_queue() const;
        [[nodiscard]] CUcontext get_command_context() const;

        void push() const;
        void pop() const;
        
    private:
        const void *command_queue_;
        mutable CUcontext context_;
  
    };
}

