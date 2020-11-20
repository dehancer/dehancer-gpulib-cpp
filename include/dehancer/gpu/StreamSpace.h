//
// Created by denn nevera on 2019-10-18.
//

#pragma once

#include "dehancer/gpu/DeviceConfig.h"
#include "dehancer/gpu/ocio/LogParams.h"
#include "dehancer/gpu/ocio/GamaParams.h"
#include "dehancer/gpu/ocio/LutParams.h"

namespace dehancer {

    namespace ocio {

        struct Params {
            LogParameters  log;
            GammaParameters gama;
        };
    }

    /***
     * We must do Camera / Film Profile transformation to WORKING space.
     * Dehancer profiles store in base Rec709 CS and 2.4 Gama
     */
    struct StreamSpace {

        typedef float4x4 Matrix;

        enum class Direction : int {
            forward = 0,
            inverse,
            none
        };

        /***
         * Color Science type
         */
        enum class Type : int {
            pass         = -1,
            color_space  = 0, // rec_709_22, aces, etc...
            camera       = 30
        };

        /***
         * Exchange format from host to gpu
         */
        struct TransformFunction {
            bool is_identity = true;

            /***
             * Forward transformation matrix from current space to another
             */
            Matrix cs_forward_matrix = get_idetntity_cs_matrix();

            /***
             * Inverse transformation matrix to current space from another
             */
            Matrix cs_inverse_matrix = get_idetntity_cs_matrix() ;

            /***
             * Polynominal and Gama transformation parameters
             */
            ocio::Params cs_params;
        };

#ifndef __METAL_VERSION__
        struct TransformLut {
            bool is_identity = true;
            ocio::LutParameters forward;
            ocio::LutParameters inverse;
        };

#endif

        /***
         * By default use rec.709 1/0.45 gama
         */
        Type                            type = Type::color_space;

        /***
         * Transform function
         */
        TransformFunction transform_function;

        /***
         * Transformed image can be analyzed and expanded
         */
        bool                       expandable = false;

#ifndef __METAL_VERSION__
        /***
         * Transform table
         */
        TransformLut           transform_lut;

        /***
         * Searchable unique id
         */
        std::string                       id = "rec_709_g22";

        /***
         * Name of space can be displayed on UI
         */
        std::string                     name = "Rec.709";
#endif

        /***
         * Default comparison operator
         * @param c - color stream space
         * @return result
         */

#ifdef __METAL_VERSION__
        bool operator==(constant StreamSpace &c) { return type == c.type; }
#else
        bool operator==(const StreamSpace &c) const { return type == c.type && id == c.id; }
#endif

#ifdef __METAL_VERSION__
        thread
#endif
        StreamSpace& operator=(
#ifdef __METAL_VERSION__
                constant
#else
                const
#endif
        StreamSpace&) = default;

        /***
         * Create default space, does nothing
         * @return identity space
         */
        inline static StreamSpace create_identity() { auto r=StreamSpace(); r.transform_function.is_identity = true; return r;};

        /***
         * Transform functor
         * @param in - input color
         * @param direction - transformation direction
         * @return new output color
         */
        float3 transform(float3 in, StreamSpace::Direction direction) const ;

        /***
        * Create new identity transformation matrix does nothing
        * @return cs matrix
        */
        inline static float4x4 get_idetntity_cs_matrix() {
          return float4x4(
                  {
                          {1.000000f, 0.000000f, 0.000000f, 0.000000f},
                          {0.000000f, 1.000000f, 0.000000f, 0.000000f},
                          {0.000000f, 0.000000f, 1.000000f, 0.000000f},
                          {0.000000f, 0.000000f, 0.000000f, 1.000000f}
                  });
        };
#ifndef __METAL_VERSION__
        /***
         * Transofrm functro
         * @param in - input vector of colors
         * @param out - new output vector of colors
         * @param direction - transformation direction
         */
        void transform(const std::vector<float3> &in, std::vector<float3> &out,
                       StreamSpace::Direction direction) const ;

        static void convert(const std::vector<float> &in, Matrix& matrix);
        static void convert(const float* in, size_t size, Matrix& matrix);
        static Matrix convert(const float* in, size_t size);
        static Matrix convert(const std::vector<float> &in);
#endif

    };
}