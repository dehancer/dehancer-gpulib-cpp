//
// Created by denn on 23.01.2021.
//

#include "dehancer/gpu/operations/GammaKernel.h"

namespace dehancer {
    GammaKernel::GammaKernel(const void *command_queue,
                             const Texture &source,
                             const Texture &destination,
                             Gamma params,
                             Direction direction,
                             float impact,
                             bool wait_until_completed, const std::string &library_path) :
            Kernel(command_queue, "kernel_gamma", source, destination, wait_until_completed, library_path),
            params_(params),
            direction_(direction),
            impact_(impact)
    {}
    
    GammaKernel::GammaKernel (const void *command_queue,
                              Gamma params,
                              Direction direction,
                              float impact,
                              bool wait_until_completed,
                              const std::string &library_path):
            GammaKernel(command_queue, nullptr, nullptr, params, direction, impact, wait_until_completed, library_path)
    {}
    
    void GammaKernel::setup (CommandEncoder &encoder) {
      encoder.set(&params_, sizeof(params_),2);
      encoder.set(&direction_, sizeof(direction_),3);
      encoder.set(impact_,4);
    }
    
    void GammaKernel::set_params (GammaKernel::Gamma params, GammaKernel::Direction direction) {
      params_ = params;
      direction_ = direction;
    }
    
    GammaKernel::Gamma GammaKernel::get_gamma () const {
      return params_;
    }
    
    GammaKernel::Direction GammaKernel::get_direction () const {
      return direction_;
    }
    
    void GammaKernel::set_gamma (GammaKernel::Gamma params) {
      params_ = params;
    }
    
    void GammaKernel::set_direction (GammaKernel::Direction direction) {
      direction_=direction;
    }
    
    float GammaKernel::get_impact () const {
      return impact_;
    }
    
    void GammaKernel::set_impact (float impact) {
      impact_ = impact;
    }
}