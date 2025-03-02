//
// Created by denn nevera on 09/06/2020.
//

#pragma once

#include "dehancer/gpu/ocio/LutParams.h"

namespace dehancer::ocio::ACES2065 {

    namespace forward {
        struct lut {
            static DHCR_LutParameters params;
        };
    }

    namespace inverse {
        struct lut {
            static DHCR_LutParameters params;
        };
    }
}