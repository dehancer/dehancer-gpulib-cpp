//
// Created by dimich on 12.01.2024.
//

#pragma once

#include <CL/cl.h>
#include <set>
#include "dehancer/Common.h"
#include "Command.h"

namespace dehancer::opencl {
    struct GPULibraryCache {
    public:
        explicit GPULibraryCache(dehancer::opencl::Command *command);

        bool has_cache(const std::string &library_source = "");

        bool compile_program(const std::string &library_source = "");

        cl_program program_for_source(const std::string &library_source, cl_device_id device_id,
                                      const std::string &p_path, const std::string &kernel_name);

    private:
        static std::mutex mutex_;
        dehancer::opencl::Command* command_;
    };
}
