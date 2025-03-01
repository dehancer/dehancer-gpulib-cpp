//
// Created by Dennis Svinarchuk on 01.03.25.
//
#include "dehancer/gpu/ocio/cs/AcesAP0.h"

namespace dehancer::ocio::AcesAP0{

  namespace forward {
    extern float __lut__data__[];
    extern size_t  __lut__size__;
    extern size_t  __lut__channels__;

    DHCR_LutParameters lut::params = {
      .enabled = true,
      .size = static_cast<uint>(forward::__lut__size__),
      .channels = static_cast<uint>(forward::__lut__channels__),
      .data = forward::__lut__data__,
};

  }

  namespace inverse {

    extern float __lut__data__[];
    extern size_t  __lut__size__;
    extern size_t  __lut__channels__;

    DHCR_LutParameters lut::params = {
      .enabled = false,
      .size = static_cast<uint>(__lut__size__),
      .channels = static_cast<uint>(__lut__channels__),
      .data = __lut__data__
};
  }}
