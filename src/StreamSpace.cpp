//
// Created by denn nevera on 03/06/2020.
//

#include "dehancer/gpu/StreamSpace.h"

namespace dehancer {
//
//    float3 StreamSpace::transform(float3 in_, StreamSpace::Direction direction) const {
//
//      float4 out = {in_[0], in_[1], in_[2], 1};
//
//      out = direction == Direction::forward ?
//            out * transform_function.cs_forward_matrix :
//            out * transform_function.cs_inverse_matrix;
//
//      float3 next = {out[0], out[1], out[2]};
//
//      if (direction == Direction::forward) {
//        if (transform_function.cs_params.log.enabled) {
//          next = dehancer::ocio::apply_log_forward(next, transform_function.cs_params.log);
//        }
//
//        if (transform_function.cs_params.gama.enabled) {
//          next = dehancer::ocio::apply_gama_forward(next, transform_function.cs_params.gama);
//        }
//      } else {
//        if (transform_function.cs_params.gama.enabled) {
//          next = dehancer::ocio::apply_gama_inverse(next, transform_function.cs_params.gama);
//        }
//
//        if (this->transform_function.cs_params.log.enabled) {
//          next = dehancer::ocio::apply_log_inverse(next, this->transform_function.cs_params.log);
//        }
//      }
//
//      out = {next[0], next[1], next[2], 1};
//
//      return {out[0], out[1], out[2]};
//    };
//
//#ifndef __METAL_VERSION__
//
//    void StreamSpace::transform(const std::vector<float3> &in, std::vector<float3> &out,
//                                StreamSpace::Direction direction) const {
//      for (const auto& v: in) {
//        out.push_back(transform(v, direction));
//      }
//    }
//
//    void StreamSpace::convert(const std::vector<float> &in, Matrix &matrix) {
//      //memcpy(&matrix, in.data(), sizeof(matrix));
//      matrix = in;
//    }
//
//    StreamSpace::Matrix dehancer::StreamSpace::convert(const std::vector<float> &in) {
//      Matrix matrix;  matrix = in;
//      //memcpy(&matrix, in.data(), sizeof(matrix));
//      return matrix;
//    }
//
//    void StreamSpace::convert(const float *in, size_t size, dehancer::StreamSpace::Matrix &matrix) {
//      //memcpy(&matrix, in, std::min(size, sizeof(matrix)));
//      size_t length = matrix.size();
//      std::vector<float> v(in,in+std::min(size,length));
//      matrix = in;
//    }
//
//    StreamSpace::Matrix dehancer::StreamSpace::convert(const float *in, size_t size) {
//      Matrix matrix;
//      size_t length = matrix.size();
//      std::vector<float> v(in,in+std::min(size,length));
//      matrix = in;
//      //memcpy(&matrix, in, std::min(size, sizeof(matrix)));
//      return matrix;
//    }
//
//#endif

}
