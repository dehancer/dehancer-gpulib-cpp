//
// Created by denn nevera on 01/12/2020.
//

#include "dehancer/gpu/operations/GaussianBlur.h"
#include "dehancer/gpu/math/ConvolveUtils.h"
#include "dehancer/gpu/Log.h"
#include <cmath>

namespace dehancer {
    
    static constexpr float MIN_DOWNSCALED_SIGMA = 2.0f;
    
    struct GaussianBlurOptions {
        std::array<float, 4> radius_array;
        float                accuracy;
    };
    
    auto kernel_blur = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
        
        data.clear();
        
        if (!user_data.has_value()) return 1.0f;
        
        auto options = std::any_cast<GaussianBlurOptions>(user_data.value());
        
        auto radius = options.radius_array.at(index);
        
        if (radius==0) return 1.0f;
        
        float sigma = radius/2.0f;
        
        int kRadius = (int)std::ceil(sigma*std::sqrt(-2.0f*std::log(options.accuracy)))+1;
        int maxRadius = (int)std::ceil(radius/2+1) * 4 - 1;

        kRadius = std::max(kRadius,maxRadius);

        auto size = kRadius;
        if (size%2==0) size+=1;
        if (size<3) size=3;
        
        bool doDownscaling = sigma > 2.0f*MIN_DOWNSCALED_SIGMA + 0.5f;
        
        int reduceBy = doDownscaling
                       ? std::min((int)std::floor(sigma/MIN_DOWNSCALED_SIGMA), size)
                       : 1;
        
        float real_sigma = doDownscaling
                           ? std::sqrt(sigma*sigma/(float)(reduceBy*reduceBy) - 1.f/3.f - 1.f/4.f)
                           : sigma;

        int new_size = size/reduceBy;
    
        dehancer::math::make_gaussian_kernel(data, new_size, real_sigma);
        
        //std::cout << " GAUSSIAN KERNEL["<<index<<"] SIZE = " << data.size() << ", origin size: " << size << " reduce: "<< reduceBy << " sigma: "<< sigma << " real sigma: "<< real_sigma<< std::endl;
    
        #ifdef PRINT_DEBUG
        dehancer::log::print(" ### #kernel_blur(base): kernel size = %i, origin size: %i, sigma: %f, reduceBy: %i", data.size(),size,sigma,reduceBy);
        #endif
        
        return 1.0f/(float)reduceBy;
    };
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                const Texture &s,
                                const Texture &d,
                                std::array<float, 4> radius,
                                const ChannelsDesc::Transform& transform,
                                DHCR_EdgeMode    edge_mode,
                                float             accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path):
            UnaryKernel(command_queue,s,d,{
                                .row = kernel_blur,
                                .col = kernel_blur,
                                .user_data = (GaussianBlurOptions){radius,accuracy_},
                                .edge_mode = edge_mode
                        },
                        transform,
                        wait_until_completed,
                        library_path)
    {
    }
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                const Texture &s,
                                const Texture &d,
                                float radius,
                                const ChannelsDesc::Transform& transform,
                                DHCR_EdgeMode edge_mode,
                                float accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path):
            GaussianBlur(command_queue,s,d,
                         {radius,radius,radius,0},
                         transform,
                         edge_mode, accuracy_,
                         wait_until_completed,
                         library_path) {
      
    }
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                std::array<float, 4> radius,
                                const ChannelsDesc::Transform& transform,
                                DHCR_EdgeMode edge_mode,
                                float accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path):
            GaussianBlur(command_queue, nullptr, nullptr,
                         radius, transform, edge_mode, accuracy_, wait_until_completed, library_path){
      
    }
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                float radius,
                                const ChannelsDesc::Transform& transform,
                                DHCR_EdgeMode edge_mode,
                                float accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path)
            :GaussianBlur(command_queue,
                          {radius,radius,radius,0},
                          transform,
                          edge_mode,accuracy_,wait_until_completed,library_path) {
      
    }
    
    void GaussianBlur::set_radius (float radius) {
      set_radius({radius,radius,radius,0});
    }
    
    void GaussianBlur::set_radius (std::array<float, 4> radius) {
      auto options = get_options();
      auto data = options.user_data.has_value()
                  ? std::any_cast<GaussianBlurOptions>(options.user_data.value())
                  : (GaussianBlurOptions){radius,GaussianBlur::accuracy};
      data.radius_array = radius;
      options.user_data = data;
      set_options(options);
    }
    
    void GaussianBlur::set_accuracy (float accuracy_) {
      auto options = get_options();
      auto data = options.user_data.has_value()
                  ? std::any_cast<GaussianBlurOptions>(options.user_data.value())
                  : (GaussianBlurOptions){{0,0,0,0},accuracy_};
      data.accuracy = accuracy_;
      options.user_data = data;
      set_options(options);
    }
    
    std::array<float, 4> GaussianBlur::get_radius () const {
      auto options = get_options();
      if (options.user_data.has_value())
        return std::any_cast<GaussianBlurOptions>(options.user_data.value()).radius_array;
      else
        return {};
    }
}