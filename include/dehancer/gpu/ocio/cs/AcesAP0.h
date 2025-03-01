//
// Created by Dennis Svinarchuk on 01.03.25.
//

#pragma once

#include "dehancer/gpu/ocio/LutParams.h"

namespace dehancer::ocio::AcesAP0{

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
