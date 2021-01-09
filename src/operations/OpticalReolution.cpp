//
// Created by denn nevera on 01/12/2020.
//

#include "dehancer/gpu/operations/OpticalReolution.h"
#include <cmath>
#include "dehancer/gpu/math/ConvolveUtils.h"

namespace dehancer {
    
    struct DeresolutiOptions {
        std::array<float, 4> radius_array;
    };
    
    auto kernel_resolution = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
        
        data.clear();
        
        auto options = std::any_cast<DeresolutiOptions>(user_data.value());
        auto radius = options.radius_array.at(index);
    
        if (radius==0) return;
        dehancer::math::magic_resampler(radius,data);
        
        std::vector<float> gaus;

        dehancer::math::make_gaussian_kernel(gaus, data.size()*2,data.size());

        float sum = 0;
        int size = data.size()/2;
        for (int i = -size; i < size; ++i) {
          data[i+size] *= gaus[i+gaus.size()/2];
          sum += data[i+size];
        }

        for (float & i : data) {
          i /= sum;
        }
        
    };
    
    OpticalReolution::OpticalReolution (const void *command_queue,
                                        const Texture &s,
                                        const Texture &d,
                                        std::array<float, 4> radius,
                                        DHCR_EdgeAddress    address_mode,
                                        bool wait_until_completed,
                                        const std::string &library_path):
            UnaryKernel(command_queue,s,d,{
                                .row = kernel_resolution,
                                .col = kernel_resolution,
                                .user_data = (DeresolutiOptions){radius},
                                .address_mode = address_mode
                        },
                        wait_until_completed,
                        library_path)
    {
    }
    
    OpticalReolution::OpticalReolution (const void *command_queue, const Texture &s, const Texture &d, float radius,
                                        DHCR_EdgeAddress address_mode, bool wait_until_completed,
                                        const std::string &library_path):
            OpticalReolution(command_queue,s,d,
                         {radius,radius,radius,0},
                         address_mode,
                         wait_until_completed,
                         library_path) {
      
    }
}