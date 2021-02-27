//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "utils.h"

namespace dehancer::cuda {

    class Context {

    public:
        Context(const void *command_queue);
        [[nodiscard]] CUstream get_command_queue() const;

    private:
        const void *command_queue_;

    };
}

