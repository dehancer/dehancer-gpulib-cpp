//
// Created by denn nevera on 10/11/2020.
//

#pragma once

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "Function.h"
#include "Context.h"

namespace dehancer::opencl {

    class CommandEncoder: public dehancer::CommandEncoder {

    public:
        explicit CommandEncoder(cl_kernel kernel, dehancer::opencl::Function* function);
        
        ~CommandEncoder() override = default;
        
        void set(const Texture &texture, int index) override;
        void set(const void *bytes, size_t bytes_length, int index) override;
        void set(const Memory& memory, int index) override;

        void set(bool p, int index) override;

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
        
        void set(const dehancer::StreamSpace &p, int index) override;
    
        cl_kernel kernel_ = nullptr;
        dehancer::opencl::Function* function_ = nullptr;
    };
}
