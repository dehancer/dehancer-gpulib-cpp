//
// Created by dimich on 12.01.2024.
//

#pragma once

#include <CL/cl.h>
#include <set>
#include "dehancer/Common.h"
#include "Command.h"

namespace dehancer::opencl {
    struct gpu_library_cache {
    public:
        bool has_cache_for_device(dehancer::opencl::Command *command, uint64_t device_id,
                                  const std::string &library_source = "");

        bool compile_program_for_device(dehancer::opencl::Command *command, uint64_t device_id,
                                        const std::string &library_source = "");

        cl_program program_for_source(cl_context context, const std::string &library_source, cl_device_id device_id,
                                      const std::string &p_path, const std::string &kernel_name);

    };

}