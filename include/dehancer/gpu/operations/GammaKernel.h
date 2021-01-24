//
// Created by denn on 23.01.2021.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/ocio/cs/Rec709.h"

namespace dehancer {
    
    /**
     * Bypass kernel.
     */
    class GammaKernel: public Kernel {
    
    public:
    
        using Gamma = DHCR_GammaParameters;
        using Direction = DHCR_TransformDirection;
        
        explicit GammaKernel(const void *command_queue,
                             const Texture &source,
                             const Texture &destination,
                             Gamma params = ocio::REC709_22::gamma_parameters,
                             Direction direction = DHCR_TransformDirection::DHCR_Forward,
                             float impact = 1.0f,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
        
        explicit GammaKernel(const void *command_queue,
                             Gamma params = ocio::REC709_22::gamma_parameters,
                             Direction direction = DHCR_TransformDirection::DHCR_Forward,
                             float impact = 1.0f,
                             bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                             const std::string &library_path = "");
        
        void setup(CommandEncoder &encoder) override;
    
        void set_params(Gamma params, Direction direction);
        void set_gamma(Gamma params);
        void set_direction(Direction direction);
        void set_impact(float impact);
        Gamma get_gamma() const;
        Direction get_direction() const;
        float get_impact() const;
        
    private:
        Gamma params_;
        Direction direction_;
        float impact_;
    };
}