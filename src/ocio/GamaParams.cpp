//
// Created by denn nevera on 04/06/2020.
//

#include "dehancer/gpu/ocio/GamaParams.h"

namespace dehancer {

    namespace ocio {

        namespace __internal__ {

            __METAL_INLINE__ float gamma_forward_channel(float x, GammaParameters params) {
                //
                // https://en.wikipedia.org/wiki/Rec._709
                //
                if (x < params.lin_side_break) return x * params.lin_side_coeff;
                return  params.lin_side_slope * pow(x , params.base) - params.lin_side_offset;
            }

            __METAL_INLINE__ float gamma_inverse_channel(float x, GammaParameters params) {
                //
                // https://en.wikipedia.org/wiki/Rec._709
                //
                if (x < params.gama_side_break) return x / params.lin_side_coeff;
                return pow((x + params.lin_side_offset) / params.lin_side_slope, 1.0f / params.base);
            }

        }

        __METAL_INLINE__ float3 apply_gama_forward(float3 in, GammaParameters params) {
            float3 out;
            out[0] = __internal__::gamma_forward_channel(in[0], params);
            out[1] = __internal__::gamma_forward_channel(in[1], params);
            out[2] = __internal__::gamma_forward_channel(in[2], params);
            return out;
        }

        __METAL_INLINE__ float3 apply_gama_inverse(float3 in, GammaParameters params) {
            float3 out;
            out[0] = __internal__::gamma_inverse_channel(in[0], params);
            out[1] = __internal__::gamma_inverse_channel(in[1], params);
            out[2] = __internal__::gamma_inverse_channel(in[2], params);
            return out;
        }
    }
}