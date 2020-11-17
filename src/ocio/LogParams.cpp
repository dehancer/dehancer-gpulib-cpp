//
// Created by denn nevera on 04/06/2020.
//

#include "dehancer/gpu/ocio/LogParams.h"

namespace dehancer {

    namespace ocio {

        namespace __internal__ {

            struct forward_m_params {
                float m_kinv;
                float m_minuskb;
                float m_minusb;
                float m_minv;
                float m_linsinv;
                float m_minuslino;
                float log_side_break;
            };

            struct inverse_m_params {
                float m_m;
                float m_b;
                float m_klog;
                float m_kb;
                float m_linb;
                float m_linearSlope;
                float m_linearOffset;
            };

            __METAL_INLINE__ float apply_log_forward_x(float in, forward_m_params params) {

                //
                // if in <= logBreak
                //  out = ( in - linearOffset ) / linearSlope
                // else
                //  out = ( pow( base, (in - logOffset) / logSlope ) - linOffset ) / linSlope;
                //
                //  out = ( exp2( log2(base)/logSlope * (in - logOffset) ) - linOffset ) / linSlope;
                //

                float out = in;

                if (in < params.log_side_break) {
                    out = params.m_linsinv * (in + params.m_minuslino);
                } else {
                    out = (in + params.m_minuskb) * params.m_kinv;
                    out = exp2(out);
                    out = (out + params.m_minusb) * params.m_minv;
                }

                return out;
            };

            __METAL_INLINE__ float apply_log_inverse_x(float in, inverse_m_params params) {

                //
                // if in <= linBreak
                //  out = linearSlope * in + linearOffset
                // else
                //  out = ( logSlope * log( base, max( minValue, (in*linSlope + linOffset) ) ) + logOffset )
                //
                //  out = log2( max( minValue, (in*linSlope + linOffset) ) ) * logSlope / log2(base) + logOffset
                //

                float out = in;

                if (in < params.m_linb) {
                    out = params.m_linearSlope * in + params.m_linearOffset;
                } else {
                    out = in * params.m_m + params.m_b;
                    out = ocio::max(FLT_MIN, out);
                    out = log2(out);
                    out = out * params.m_klog + params.m_kb;
                }

                return out;
            }
        }

        __METAL_INLINE__ float3 apply_log_forward(float3 in, LogParameters params) {
            __internal__::forward_m_params m;
            m.m_kinv = params.log2_base / params.log_side_slope;
            m.m_minuskb = -params.log_side_offset;
            m.m_minusb = -params.lin_side_offset;
            m.m_minv = 1.0f / params.lin_side_slope;
            m.m_linsinv = 1.0f / params.linear_slope;
            m.m_minuslino = -params.linear_offset;

            float3 out;
            out[0] = apply_log_forward_x(in[0], m);
            out[1] = apply_log_forward_x(in[1], m);
            out[2] = apply_log_forward_x(in[2], m);

            return out;
        }

        __METAL_INLINE__ float3 apply_log_inverse(float3 in, LogParameters params) {
            __internal__::inverse_m_params m;

            m.m_m = params.lin_side_slope;
            m.m_b = params.lin_side_offset;
            m.m_klog = params.log_side_slope / params.log2_base;
            m.m_kb = params.log_side_offset;
            m.m_linb = params.lin_side_break;
            m.m_linearSlope = params.linear_slope;
            m.m_linearOffset = params.linear_offset;

            float3 out;
            out[0] = apply_log_inverse_x(in[0], m);
            out[1] = apply_log_inverse_x(in[1], m);
            out[2] = apply_log_inverse_x(in[2], m);

            return out;
        }
    }
}