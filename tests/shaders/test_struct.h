//
// Created by denn on 23.01.2021.
//

#ifndef DEHANCER_GPULIB_TEST_STRUCT_H
#define DEHANCER_GPULIB_TEST_STRUCT_H

#include "dehancer/gpu/kernels/common.h"
#include "dehancer/gpu/kernels/stream_space.h"

//#include "dehancer/gpu/spaces/StreamTransform.h"


typedef struct  {
//#if defined(CL_VERSION_1_2)
//    float4x4      mat;
//#else
//  dehancer::math::float4x4 mat;
//#endif
    DHCR_TransformDirection direction;
    DHCR_TransformType transform_type;
    bool_t  enabled;
    int      size;
    float   data;
} TestStruct;

#endif //DEHANCER_GPULIB_TEST_STRUCT_H
