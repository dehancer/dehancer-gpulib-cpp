#include "dehancer/gpu/ocio/cs/DVRWGRec709.h"

namespace dehancer::ocio::DVRWGRec709{

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
                .enabled = true,
                .size = static_cast<uint>(__lut__size__),
                .channels = static_cast<uint>(__lut__channels__),
                .data = __lut__data__
        };
    }}