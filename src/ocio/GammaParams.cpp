//
// Created by denn nevera on 04/06/2020.
//

#include "dehancer/gpu/ocio/GammaParams.h"
#include "dehancer/gpu/kernels/gamma.h"
#include <cmath>

namespace dehancer::ocio {

        namespace _internal_ {

            inline float gamma_forward_channel(float x, GammaParameters params) {
                //
                // https://en.wikipedia.org/wiki/Rec._709
                //
                if (x < params.lin_side_break) return x * params.lin_side_coeff;
                return  params.lin_side_slope * std::pow(x , params.base) - params.lin_side_offset;
            }
    
            inline  float gamma_inverse_channel(float x, GammaParameters params) {
                //
                // https://en.wikipedia.org/wiki/Rec._709
                //
                if (x < params.gama_side_break) return x / params.lin_side_coeff;
                return std::pow((x + params.lin_side_offset) / params.lin_side_slope, 1.0f / params.base);
            }

        }
    
        inline  float3 apply_gama_forward(float3 in, GammaParameters params) {
            float3 out;
            out[0] = _internal_::gamma_forward_channel(in[0], params);
            out[1] = _internal_::gamma_forward_channel(in[1], params);
            out[2] = _internal_::gamma_forward_channel(in[2], params);
            return out;
        }
    
        inline  float3 apply_gama_inverse(float3 in, GammaParameters params) {
            float3 out;
            out[0] = _internal_::gamma_inverse_channel(in[0], params);
            out[1] = _internal_::gamma_inverse_channel(in[1], params);
            out[2] = _internal_::gamma_inverse_channel(in[2], params);
            return out;
        }
    }