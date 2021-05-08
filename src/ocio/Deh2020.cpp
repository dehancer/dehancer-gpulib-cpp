//
// Created by denn nevera on 09/06/2020.
//

#include "dehancer/gpu/ocio/cs/Deh2020.h"

namespace dehancer::ocio::DEH2020 {

    namespace forward {
        extern float __lut__data__[];
        extern uint  __lut__size__;
        extern uint  __lut__channels__;

        LutParameters lut::params = {
                .enabled = true,
                .size = forward::__lut__size__,
                .channels = forward::__lut__channels__,
                .data = forward::__lut__data__,
        };

    }

    namespace inverse {

        extern float __lut__data__[];
        extern uint  __lut__size__;
        extern uint  __lut__channels__;

        LutParameters lut::params = {
                .enabled = true,
                .size = __lut__size__,
                .channels = __lut__channels__,
                .data = __lut__data__
        };
    }
}