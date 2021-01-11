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
    
        if (!user_data.has_value()) return ;
    
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
                      DHCR_EdgeMode    edge_mode,
                      bool wait_until_completed,
                      const std::string &library_path):
            UnaryKernel(command_queue,s,d,{
                                .row = kernel_box_blur,
                                .col = kernel_box_blur,
                                .user_data = (BoxBlurOptions){radius},
                                .edge_mode = edge_mode
                        },
                        wait_until_completed,
                        library_path)
    {
    }
    
    BoxBlur::BoxBlur (const void *command_queue,
                      const Texture &s,
                      const Texture &d,
                      size_t radius,
                      DHCR_EdgeMode edge_mode,
                      bool wait_until_completed,
                      const std::string &library_path):
            BoxBlur(command_queue,s,d,
                    {radius,radius,radius,0},
                    edge_mode,
                    wait_until_completed,
                    library_path) {
      
    }
    
    BoxBlur::BoxBlur (const void *command_queue,
                      std::array<size_t, 4> radius,
                      DHCR_EdgeMode edge_mode,
                      bool wait_until_completed,
                      const std::string &library_path):
            BoxBlur(command_queue, nullptr, nullptr,
                    radius, edge_mode, wait_until_completed, library_path){
      
    }
    
    BoxBlur::BoxBlur (const void *command_queue,
                      size_t radius,
                      DHCR_EdgeMode edge_mode,
                      bool wait_until_completed,
                      const std::string &library_path):
            BoxBlur(command_queue,{radius,radius,radius,0},edge_mode,wait_until_completed,library_path){
      
    }
    
    void BoxBlur::set_radius (std::array<size_t , 4> radius) {
      auto options = get_options();
      auto data = options.user_data.has_value()
                  ? std::any_cast<BoxBlurOptions>(options.user_data.value())
                  : (BoxBlurOptions){radius};
      data.radius_array = radius;
      set_user_data(data);
    }
    
    void BoxBlur::set_radius (size_t radius) {
      set_radius({radius,radius,radius,0});
    }
}