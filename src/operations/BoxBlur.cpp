//
// Created by denn nevera on 01/12/2020.
//

#include "dehancer/gpu/operations/BoxBlur.h"

#include <cmath>
#include "dehancer/gpu/math/ConvolveUtils.h"

namespace dehancer {
    
    struct BoxBlurOptions {
        std::array<size_t, 4> radius_array;
    };
    
    auto kernel_box_blur = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
        
        data.clear();
        
        auto options = std::any_cast<BoxBlurOptions>(user_data.value());
        
        auto radius = options.radius_array.at(index);
    
        if (radius <= 1 ) return;
        for (int i = 0; i < radius; ++i) {
          data.push_back(1.0f/(float)radius);
        }
    };
    
    BoxBlur::BoxBlur (const void *command_queue,
                      const Texture &s,
                      const Texture &d,
                      std::array<size_t, 4> radius,
                      DHCR_EdgeAddress    address_mode,
                      bool wait_until_completed,
                      const std::string &library_path):
            UnaryKernel(command_queue,s,d,{
                                .row = kernel_box_blur,
                                .col = kernel_box_blur,
                                .user_data = (BoxBlurOptions){radius},
                                .address_mode = address_mode
                        },
                        wait_until_completed,
                        library_path)
    {
    }
    
    BoxBlur::BoxBlur (const void *command_queue, const Texture &s, const Texture &d, size_t radius,
                      DHCR_EdgeAddress address_mode, bool wait_until_completed,
                      const std::string &library_path):
            BoxBlur(command_queue,s,d,
                         {radius,radius,radius,0},
                         address_mode,
                         wait_until_completed,
                         library_path) {
      
    }
}