//
// Created by dimich on 12.01.2024.
//

#ifndef DEHANCER_GPULIB_OPENCLCACHE_H
#define DEHANCER_GPULIB_OPENCLCACHE_H

#include <CL/cl.h>
#include <set>
#include "dehancer/Common.h"

namespace dehancer::opencl {
    using oclProgramCompileHandlerDecl = void (*)(const bool started, const std::string &device_name, void *context);

    class OpenCLCache : public Singleton<OpenCLCache> {
    public:
        cl_program program_for_source(cl_context context, const std::string &library_source, cl_device_id device_id,
                                      const std::string &p_path, const std::string &kernel_name);

        void add_compile_handler(oclProgramCompileHandlerDecl, void *context);

        void remove_compile_handler(oclProgramCompileHandlerDecl);

    private:
        void notify_compile_handlers(const bool started, const std::string &device_name);

    private:
        std::mutex _handlers_mutex;
        std::set<std::pair<void *, oclProgramCompileHandlerDecl>> _compile_handlers;

    };

}

#endif //DEHANCER_GPULIB_OPENCLCACHE_H
