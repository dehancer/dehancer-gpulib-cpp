//
// Created by denn nevera on 20/11/2020.
//

#include "dehancer/gpu/CommandEncoder.h"

namespace dehancer {

    void dehancer::CommandEncoder::set(const float2 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void CommandEncoder::set(const float3 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void CommandEncoder::set(const float4 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void dehancer::CommandEncoder::set(const float2x2 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void CommandEncoder::set(const float3x3 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }

    void CommandEncoder::set(const float4x4 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }
}