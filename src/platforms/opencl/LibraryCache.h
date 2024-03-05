//
// Created by dimich on 12.01.2024.
//

#pragma once

#include <CL/cl.h>
#include <set>
#include "dehancer/Common.h"
#include "Command.h"

namespace dehancer::opencl {
    class GPULibraryCache {
    public:
        explicit GPULibraryCache(dehancer::opencl::Command *command);

        bool exists(const std::string &library_source = "");

        bool compile_program(const std::string &library_source = "");

        cl_program program_for_source(const std::string &library_source,
                                      const std::string &p_path, const std::string &kernel_name);

    private:
        [[nodiscard]] std::string get_cache_file_name(const std::string &library_source) const;

        [[nodiscard]] std::string get_device_name() const;

    private:
        static std::mutex mutex_;
        dehancer::opencl::Command *command_;
    };
}
