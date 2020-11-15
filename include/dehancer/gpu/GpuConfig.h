//
// Created by denn nevera on 2019-07-21.
//

#include "dehancer/gpu/GpuTypedefs.h"

#if defined(DEHANCER_GPU_METAL)

#elif defined(DEHANCER_GPU_OPENCL)

#else

#error "You must define GPU Layer"

#endif