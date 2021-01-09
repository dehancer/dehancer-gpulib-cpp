//
// Created by denn nevera on 01/12/2020.
//

#pragma once

//#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/operations/UnaryKernel.h"

namespace dehancer {

    class GaussianBlur: public UnaryKernel {
    public:

        GaussianBlur(const void* command_queue,
                     const Texture&    s,
                     const Texture&    d,
                     std::array<int,4> radius,
                     EdgeAddress       address_mode = EdgeAddress::ADDRESS_CLAMP,
                     float             accuracy = 0.001,
                     bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                     const std::string& library_path = ""
        );
        
//    private:
//        int box_number_ = 3;
//        std::array<int,4> radius_;
//        std::array<std::vector<float>,4> radius_boxes_;
//        size_t w_;
//        size_t h_;
//        Channels channels_out_;
//        ChannelsOutput channels_finalizer_;
    };
}

