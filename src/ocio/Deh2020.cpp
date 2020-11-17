//
// Created by denn nevera on 09/06/2020.
//

#include "dehancer/gpu/ocio/cs/Deh2020.h"

namespace dehancer::ocio::DEH2020 {

    namespace forward {
        extern float __lut__data__[];
        extern size_t __lut__size__;
        extern size_t __lut__channels__;

        LutParameters lut::params = {
                .data = forward::__lut__data__,
                .size = forward::__lut__size__,
                .channels = forward::__lut__channels__,
                .enabled = true
        };

    }

    namespace inverse {

        extern float __lut__data__[];
        extern size_t __lut__size__;
        extern size_t __lut__channels__;

        LutParameters lut::params = {
                .data = __lut__data__,
                .size = __lut__size__,
                .channels = __lut__channels__,
                .enabled = true
        };
    }
}