#pragma once 

#include "dehancer/gpu/ocio/LutParams.h"

namespace dehancer::ocio::DVRWGRec709{

namespace forward {
      struct lut {
            static DHCR_LutParameters params;
      };
}

namespace inverse {
        struct lut {
            static DHCR_LutParameters params;
        };
    };
}

