//
// Created by denn nevera on 2019-07-23.
//

#include "dehancer/gpu/clut/CLutHaldIdentity.h"

namespace dehancer {

    CLutHaldIdentity::CLutHaldIdentity(
            const void *command_queue,
            uint lut_size,
            bool wait_until_completed,
            const std::string &library_path):
            Function(command_queue, "kernel_make2DHaldLutBuffer", wait_until_completed, library_path),
            CLut(),
            level_((size_t)std::sqrtf((float)lut_size)),
            lut_size_(lut_size)
    {

        auto size = level_*level_*level_;
        texture_ = make_texture(size,size);

//        auto device = get_device();
//
//        auto buffer = [device newBufferWithLength:lut_size_*lut_size_*lut_size_*sizeof(float)*4
//                                          options:MTLResourceStorageModeShared];
//
//        auto w = 4;
//        compute_size_.threadsPerThreadgroup = MTLSizeMake(
//                w,
//                w,
//                w);
//
//        compute_size_.threadGroups =  MTLSizeMake(
//                (lut_size_+w)/w,
//                (lut_size_+w)/w,
//                (lut_size_+w)/w);
//
//        auto denom = float(lut_size_ - 1);
//
//        execute([this,buffer,denom,size](id<MTLComputeCommandEncoder>& compute_encoder) {
//
//            [compute_encoder setBytes:&lut_size_ length:sizeof(lut_size_) atIndex:0];
//            [compute_encoder setBytes:&denom length:sizeof(denom) atIndex:1];
//
//            [compute_encoder setBuffer:buffer offset:0 atIndex:2];
//
//            return texture_;
//        });

/*
        auto p = (float*)[buffer contents];
        auto denom = float(lut_size_ - 1);

        for(uint i = 0; i < lut_size_; i++)
        {
            for(uint j = 0; j < lut_size_; j++)
            {
                for(uint k = 0; k < lut_size_; k++)
                {
                    *p++ = (float)k / denom;
                    *p++ = (float)j / denom;
                    *p++ = (float)i / denom;
                    *p++ = 1;
                }
            }
        }
*/
//        auto queue = get_command_queue();
//
//        id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
//
//        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
//
//        [blitEncoder copyFromBuffer:buffer
//                       sourceOffset:0
//                  sourceBytesPerRow: size*sizeof(float)*4
//                sourceBytesPerImage: size*size*sizeof(float)*4
//                         sourceSize: MTLSizeMake(texture_.width,texture_.height,1)
//                          toTexture: texture_
//                   destinationSlice: 0
//                   destinationLevel: 0
//                  destinationOrigin: MTLOriginMake(0,0,0)];
//
//        [blitEncoder endEncoding];
//
//        [commandBuffer commit];
//
//        if (wait_until_completed_ || WAIT_UNTIL_COMPLETED)
//            [commandBuffer waitUntilCompleted];
//
//        [buffer release];
    }
    
}