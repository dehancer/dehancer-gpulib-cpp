//
// Created by denn nevera on 18/11/2020.
//

#pragma once

#include "dehancer/gpu/Memory.h"
#include "Context.h"

namespace dehancer::cuda {

    struct MemoryHolder: public dehancer::MemoryHolder, public cuda::Context {
        MemoryHolder(const void *command_queue, const void* buffer, size_t length, MemoryDesc::MemFlags flags=MemoryDesc::MemFlags::read_write);
        MemoryHolder(const void *command_queue, std::vector<uint8_t> buffer);
        MemoryHolder(const void *command_queue, void* device_memory);
        ~MemoryHolder() override ;

        size_t get_length() const override;
        [[nodiscard]] const void*  get_memory() const override;
        [[nodiscard]] void*  get_memory() override;

        const void * get_pointer() const override;
        void * get_pointer() override;

        Error get_contents(std::vector<uint8_t>& buffer) const override;
        Error get_contents(void *buffer, size_t length) const override;

    private:
        CUdeviceptr memobj_;
        size_t length_;
        bool is_self_allocated_;
    };
}

