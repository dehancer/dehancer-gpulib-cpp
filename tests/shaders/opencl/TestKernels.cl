//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/kernels/opencl/opencl.h"
#include "aoBenchKernel.h"
#include "TestKernels.h"


////@formatter:off
//__DEHANCER_KERNEL__ void convolve_horizontal_image_kernel(
//        __read_only image2d_t source BIND_TEXTURE(1),
//        __write_only image2d_t destination BIND_TEXTURE(2),
//        __DEHANCER_DEVICE_ARG__ float* weights BIND_BUFFER(3),
//        int size BIND_BUFFER(4)
//) {
//
//int x = get_global_id(0);
//int y = get_global_id(1);
//int w = get_image_width(source);
//int h = get_image_height(source);
//
//int2 gid = (int2)(x, y);
//
//if ((gid.x < w) && (gid.y < h)) {
//float4 color = (float4)(0,0,0,1);
//for (int i = -size/2; i < size/2; ++i) {
//int2 gidx = gid;
//gidx.x += i;
//if (gidx.x<0) continue;
//if (gidx.x>=w) continue;
//
//color += read_imagef(source, nearest_sampler, gidx) * weights[i];
//
//}
//
//write_imagef(destination, gid, color);
//}
//}
////@formatter:on
