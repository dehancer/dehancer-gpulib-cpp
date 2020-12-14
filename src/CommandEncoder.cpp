//
// Created by denn nevera on 20/11/2020.
//

#include "dehancer/gpu/CommandEncoder.h"

namespace dehancer {

    void dehancer::CommandEncoder::set(bool p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(char p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(int8_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(int16_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(int32_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(uint8_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(uint16_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(uint32_t p, int index){
      set(&p, sizeof(p), index);
    }

    void dehancer::CommandEncoder::set(float p, int index) {
      set(&p, sizeof(p), index);
    }

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

    void CommandEncoder::set(const float4x4 &p, int index) {
      set(p.mem, p.size()*sizeof(float), index);
    }
}