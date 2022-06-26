//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "Function.h"
#include "Context.h"
#include <any>

#include "dehancer/gpu/kernels/cuda/texture2d.h"

namespace dehancer::cuda {

    class CommandEncoder: public dehancer::CommandEncoder {

    public:
        explicit CommandEncoder(const CUfunction kernel, const dehancer::cuda::Function* function);
        void set(const Texture &texture, int index) override;
        void set(const void *bytes, size_t bytes_length, int index) override;
        void set(const Memory& memory, int index) override;

        void set(bool p, int index) override;
        void set(int p, int index) override;

        void set(float p, int index) override;
        void set(const float2& p, int index) override;
        void set(const float3& p, int index) override;
        void set(const float4& p, int index) override;
    
        void set(const float2x2& m, int index) override;
        void set(const float3x3& m, int index) override;
        void set(const float4x4& m, int index) override;
    
        void set(const math::uint2& p, int index) override;
        void set(const math::uint3& p, int index) override;
        void set(const math::uint4& p, int index) override;
    
        void set(const math::int2& p, int index) override;
        void set(const math::int3& p, int index) override;
        void set(const math::int4& p, int index) override;
    
        void set(const math::bool2& p, int index) override;
        void set(const math::bool3& p, int index) override;
        void set(const math::bool4& p, int index) override;
        
        [[nodiscard]] size_t get_block_max_size() const override;
        [[nodiscard]] ComputeSize ask_compute_size(size_t width, size_t height, size_t depth) const override;
    
        CUfunction kernel_ = nullptr;
        mutable dehancer::cuda::Function* function_ = nullptr;
        std::vector<void* > args_;
        std::vector<std::any> args_container_;

        void resize_at_index(int index);

        ~CommandEncoder() override = default;
        
    };
}
