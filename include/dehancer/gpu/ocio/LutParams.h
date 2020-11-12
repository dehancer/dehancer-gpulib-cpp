//
// Created by denn nevera on 09/06/2020.
//

#pragma once

#include "Params.h"

namespace dehancer {

    namespace ocio {
        ///
        /// Only host system is supported now
        ///
#ifndef __METAL_VERSION__
        struct LutParameters {
            float* data{};
            size_t size{};
            size_t channels{};
            bool  enabled = false;
        };
#endif
    }
}