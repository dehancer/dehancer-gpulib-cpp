//
// Created by dimich on 12.01.2024.
//

#include <stdexcept>
#include "LibraryCache.h"
#include <iomanip>
#include <sstream>
#include <fstream>
#include "dehancer/gpu/Log.h"
#include "dehancer/gpu/DeviceCache.h"
#include "dehancer/gpu/Paths.h"
#include "dehancer/opencl/embeddedProgram.h"

namespace dehancer::opencl {

    cl_program gpu_library_cache::program_for_source(cl_context context, const std::string &library_source,
                                                const cl_device_id device_id, const std::string &p_path,
                                                const std::string &kernel_name) {
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

        return program;
    }


    bool gpu_library_cache::has_cache_for_device(dehancer::opencl::Command *command, const uint64_t device_id,
                                            const std::string &library_source) {
        auto device = static_cast<cl_device_id>( DeviceCache::Instance().get_device(device_id));

        if (device == nullptr) {
            return false;
        }

        char cBuffer[1024];
        char *cBufferN;

        cl_int err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);

        if (err != CL_SUCCESS) {
            return false;
        }
        std::string device_name(cBuffer);
        auto p_path = dehancer::device::get_lib_path();
        std::string source = (library_source.empty()) ? clHelper::getEmbeddedProgram(p_path) : library_source;

    }

    bool gpu_library_cache::compile_program_for_device(dehancer::opencl::Command *command, const uint64_t device_id,
                                                  const std::string &library_source) {
        auto device = static_cast<cl_device_id>( DeviceCache::Instance().get_device(device_id));

        if (device == nullptr) {
            return false;
        }

        std::string device_name = device::get_name(device);

        if(device_name.empty()) {
            return false;
        }

        auto p_path = dehancer::device::get_lib_path();
        std::string source = (library_source.empty()) ? clHelper::getEmbeddedProgram(p_path) : library_source;

        const char *source_str = library_source.c_str();
        size_t source_size = library_source.size();

        cl_int err;
        cl_program program = clCreateProgramWithSource(command->get_context(), 1, (const char **) &source_str,
                                                       (const size_t *) &source_size, &err);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
        }

        err = clBuildProgram(program, 1, &device,
                             "-cl-std=CL2.0 -cl-kernel-arg-info -cl-unsafe-math-optimizations -cl-single-precision-constant",
                             nullptr, nullptr);


        size_t n;
        clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, 0, nullptr, &n);
        size_t sizes[n];
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, n * sizeof(size_t), sizes, nullptr);

        auto **binaries = new unsigned char *[n];
        for (int i = 0; i < (int) n; ++i) {
            binaries[i] = new unsigned char[sizes[i]];
        }
        clGetProgramInfo(program, CL_PROGRAM_BINARIES, n * sizeof(unsigned char *), binaries, nullptr);
        auto cache_path = dehancer::device::get_opencl_cache_path();

        auto h1 = std::hash<std::string>{}(library_source);
        auto h2 = std::hash<std::string>{}(device_name);

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(sizeof(size_t) * 2) << std::hex << h1 << "_" << h2;
        std::string cache_file_name(ss.str());
        {
            std::ofstream ostrm(cache_path + cache_file_name, std::ios::binary);
            ostrm.write(reinterpret_cast<char*>(&binaries[0]), sizeof source_size);
        }

        return true;
    }


}