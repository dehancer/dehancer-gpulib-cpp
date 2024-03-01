//
// Created by dimich on 12.01.2024.
//

#include <stdexcept>
#include "OpenCLCache.h"
#include <iomanip>
#include <sstream>

namespace dehancer::opencl {

    cl_program OpenCLCache::program_for_source(cl_context context, const std::string& library_source, const cl_device_id device_id) {
        if(device_id == nullptr) {
            return nullptr;
        }

        char cBuffer[1024];
        char *cBufferN;
        cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);

        if(err != CL_SUCCESS) {
            return nullptr;
        }

        std::string device_name(cBuffer);
        auto h1 = std::hash<std::string>{}(library_source);
        auto h2= std::hash<std::string>{}(device_name);

        std::stringstream ss;
        ss <<  std::setfill ('0') << std::setw(sizeof(size_t)*2) << std::hex << h1 << "_" << h2;
        std::string cache_file_name(ss.str());

        //test cache_file_existst here

        //else create from source
        const char *source_str = library_source.c_str();
        size_t source_size = library_source.size();

        notify_compile_handlers(true, device_name);

        cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str,
                                             (const size_t *) &source_size, &err);

        if(err != CL_SUCCESS) {
            throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
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