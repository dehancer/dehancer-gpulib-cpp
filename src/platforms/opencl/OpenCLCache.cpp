//
// Created by dimich on 12.01.2024.
//

#include <stdexcept>
#include "OpenCLCache.h"
#include <iomanip>
#include <sstream>
#include "dehancer/gpu/Log.h"

namespace dehancer::opencl {

    cl_program OpenCLCache::program_for_source(cl_context context, const std::string &library_source,
                                               const cl_device_id device_id, const std::string& p_path, const std::string& kernel_name) {
        if (device_id == nullptr) {
            return nullptr;
        }

        char cBuffer[1024];
        char *cBufferN;
        cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);

        if (err != CL_SUCCESS) {
            return nullptr;
        }

        std::string device_name(cBuffer);
        auto h1 = std::hash<std::string>{}(library_source);
        auto h2 = std::hash<std::string>{}(device_name);

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(sizeof(size_t) * 2) << std::hex << h1 << "_" << h2;
        std::string cache_file_name(ss.str());

        //test cache_file_existst here

        //else create from source
        const char *source_str = library_source.c_str();
        size_t source_size = library_source.size();

        notify_compile_handlers(true, device_name);

        cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str,
                                                       (const size_t *) &source_size, &err);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
        }

        err = clBuildProgram(program, 1, &device_id,
                             "-cl-std=CL2.0 -cl-kernel-arg-info -cl-unsafe-math-optimizations -cl-single-precision-constant",
                             nullptr, nullptr);

        if (err != CL_SUCCESS) {

            std::string log = "Unable to build OpenCL program from: " + kernel_name;

            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
            log.resize(log_size);

            // Get the log
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  log_size, log.data(), nullptr);

            log::error(true, "OpenCL Function build Error[%i]: %s", err, log.c_str());
            throw std::runtime_error(
                    "Unable to build OpenCL program from: '" + p_path + "' on: " + kernel_name + ": \n[" +
                    std::to_string(log_size) + "] " + log);
        }

        notify_compile_handlers(false, device_name);

        return program;
    }

    void OpenCLCache::add_compile_handler(oclProgramCompileHandlerDecl handler, void *context) {
        std::lock_guard<std::mutex> g(_handlers_mutex);
        _compile_handlers.insert({context, handler});
    }

    void OpenCLCache::remove_compile_handler(oclProgramCompileHandlerDecl handler) {
        std::lock_guard<std::mutex> g(_handlers_mutex);
        auto it = std::find_if(_compile_handlers.begin(), _compile_handlers.end(),
                               [&handler](auto pair) {
                                   return handler == pair.second;
                               });
        if (it != _compile_handlers.end()) {
            _compile_handlers.erase(it);
        }
    }

    void OpenCLCache::notify_compile_handlers(const bool started, const std::string &device_name) {
        for (auto handler: _compile_handlers) {
            handler.second(started, device_name, handler.first);
        }
    }
}