//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "Function.h"
#include "Context.h"
#include <any>

namespace dehancer::cuda {

    class CommandEncoder: public dehancer::CommandEncoder {

    public:
        explicit CommandEncoder(CUfunction kernel, dehancer::cuda::Function* function);
        void set(const Texture &texture, int index) override;
        void set(const void *bytes, size_t bytes_length, int index) override;
        void set(const Memory& memory, int index) override;

        void set(bool p, int index) override;

        void set(float p, int index) override;
        void set(const float2& p, int index) override;
        void set(const float3& p, int index) override;
        void set(const float4& p, int index) override;

        void set(const float2x2& m, int index) override;
        void set(const float4x4& m, int index) override;

        dehancer::cuda::Function* function_ = nullptr;
        CUfunction kernel_ = nullptr;
        std::vector<void* > args_;
        std::vector<std::any> args_container_;

        void resize_at_index(int index);

    };
}
