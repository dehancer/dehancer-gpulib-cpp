//
// Created by denn nevera on 01/12/2020.
//

#include "dehancer/gpu/operations/OpticalResolution.h"
#include "dehancer/gpu/math/ConvolveUtils.h"
#include <cmath>

namespace dehancer {
    
    struct DeresolutionOptions {
        std::array<float, 4> radius_array;
    };
    
    auto kernel_resolution = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
        
        data.clear();
    
        if (!user_data.has_value()) return 1.0f;
    
        auto options = std::any_cast<DeresolutionOptions>(user_data.value());
        auto radius = options.radius_array.at(index);
    
        if (radius==0) return 1.0f;
        if (radius>3.0f)
          dehancer::math::magic_resampler(radius,data);
        else {
          data.push_back(1);
          data.push_back(1);
          data.push_back(1);
        }
        
        std::vector<float> gauss;
        
        size_t kernel_size = data.size();
        auto s_radius = static_cast<int>(kernel_size/2);
        auto sigma = static_cast<float>(s_radius);
        dehancer::math::make_gaussian_kernel(gauss, kernel_size, sigma);

        float sum = 0;
        int size = s_radius;
        for (int i = -size; i <= size; ++i) {
          auto ni = i+size;
          data[ni] *= gauss[ni];
          sum += data[ni];
        }

        for (float & i : data) {
          i /= sum;
        }

        std::cout << " ====== " << data.size() << std::endl;
        for (int i = 0; i < data.size(); ++i) {
          std::cout << "optical resolution["<<i<<"]: = " << data[i]  << "" << std::endl;
        }
        std::cout << " /====== \n" << std::endl;

        return 1.0f;
    };
    
    OpticalResolution::OpticalResolution (const void *command_queue,
                                          const Texture &s,
                                          const Texture &d,
                                          std::array<float, 4> radius,
                                          const ChannelsDesc::Transform& transform,
                                          DHCR_EdgeMode    edge_mode,
                                          bool wait_until_completed,
                                          const std::string &library_path):
            UnaryKernel(command_queue,s,d,{
                                .row = kernel_resolution,
                                .col = kernel_resolution,
                                .user_data = (DeresolutionOptions){radius},
                                .edge_mode = edge_mode
                        },
                        transform,
                        wait_until_completed,
                        library_path)
    {
    }
    
    OpticalResolution::OpticalResolution (const void *command_queue,
                                          const Texture &s,
                                          const Texture &d,
                                          float radius,
                                          const ChannelsDesc::Transform& transform,
                                          DHCR_EdgeMode address_mode,
                                          bool wait_until_completed,
                                          const std::string &library_path):
            OpticalResolution(command_queue, s, d,
                              {radius,radius,radius,0},transform,
                              address_mode,
                              wait_until_completed,
                              library_path) {
      
    }
    
    void OpticalResolution::set_radius (float radius) {
      set_radius({radius,radius,radius,0});
    }
    
    void OpticalResolution::set_radius (std::array<float, 4> radius) {
      set_user_data((DeresolutionOptions){radius});
    }
    
    OpticalResolution::OpticalResolution (const void *command_queue,
                                          std::array<float, 4> radius,
                                          const ChannelsDesc::Transform& transform,
                                          DHCR_EdgeMode edge_mode,
                                          bool wait_until_completed,
                                          const std::string &library_path):
            OpticalResolution(command_queue, nullptr, nullptr, radius, transform, edge_mode, wait_until_completed, library_path){
      
    }
    
    OpticalResolution::OpticalResolution (const void *command_queue,
                                          float radius,
                                          const ChannelsDesc::Transform& transform,
                                          DHCR_EdgeMode edge_mode,
                                          bool wait_until_completed, const std::string &library_path):
            OpticalResolution(command_queue, {radius, radius, radius, 0}, transform, edge_mode, wait_until_completed, library_path){
      
    }
}