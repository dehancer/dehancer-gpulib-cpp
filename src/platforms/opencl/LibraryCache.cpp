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
    std::mutex GPULibraryCache::mutex_;

    GPULibraryCache::GPULibraryCache(dehancer::opencl::Command *command)
            : command_(command) {

    }


    cl_program GPULibraryCache::program_for_source(const std::string &library_source,
                                                   const std::string &p_path,
                                                   const std::string &kernel_name) {
        std::unique_lock<std::mutex> lock(GPULibraryCache::mutex_);

        auto device_id = command_->get_device_id();
        cl_int err;
        cl_program program = nullptr;

        if (exists(library_source)) {
            cl_int bin_status;

            auto cache_path = dehancer::device::get_opencl_cache_path();
            auto cache_file_name = get_cache_file_name(library_source);
            {
                std::ifstream file((cache_path + "/" + cache_file_name).c_str(), std::ios::binary);
                auto data = std::vector<unsigned char>((std::istreambuf_iterator<char>(file)),
                                                       std::istreambuf_iterator<char>());

                size_t size = data.size();
                unsigned char *data_ptr = data.data();

                program = clCreateProgramWithBinary(command_->get_context(), 1, &device_id,
                                                    (const size_t *) &size,
                                                    (const unsigned char **) &data_ptr, &bin_status, &err);
            }
        } else {

            //else create from source
            const char *source_str = library_source.c_str();
            size_t source_size = library_source.size();

            program = clCreateProgramWithSource(command_->get_context(), 1, (const char **) &source_str,
                                                (const size_t *) &source_size, &err);
        }

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


    bool GPULibraryCache::exists(const std::string &library_source) {

        auto device = command_->get_device_id();

        if (device == nullptr) {
            return false;
        }

        auto p_path = dehancer::device::get_lib_path();
        std::string source = (library_source.empty()) ? clHelper::getEmbeddedProgram(p_path) : library_source;

        auto cache_path = dehancer::device::get_opencl_cache_path();
        std::string cache_file_name = get_cache_file_name(library_source);

        std::ifstream f((cache_path + "/" + cache_file_name).c_str());
        return f.good();
    }

    std::string GPULibraryCache::get_device_name() const {
        char cBuffer[1024];
        char *cBufferN;

        cl_int err = clGetDeviceInfo(command_->get_device_id(), CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Unable to get OpenCL device info");
        }

        return {cBuffer};
    }

    std::string GPULibraryCache::get_cache_file_name(const std::string &library_source) const {
        auto cache_path = dehancer::device::get_opencl_cache_path();

        auto h1 = std::hash<std::string>{}(library_source);
        auto h2 = std::hash<std::string>{}(get_device_name());

        std::stringstream ss;
        ss << std::setfill('0') << std::setw(sizeof(size_t) * 2) << std::hex << h1 << "_" << h2;
        return std::string(ss.str());
    }

    bool GPULibraryCache::compile_program(const std::string &library_source) {

        std::unique_lock<std::mutex> lock(GPULibraryCache::mutex_);

        auto device = command_->get_device_id();

        if (device == nullptr) {
            return false;
        }

        auto p_path = dehancer::device::get_lib_path();
        std::string source = (library_source.empty()) ? clHelper::getEmbeddedProgram(p_path) : library_source;

        const char *source_str = library_source.c_str();
        size_t source_size = library_source.size();

        cl_int err;
        cl_program program = clCreateProgramWithSource(command_->get_context(), 1, (const char **) &source_str,
                                                       (const size_t *) &source_size, &err);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
        }

        err = clBuildProgram(program, 1, &device,
                             "-cl-std=CL2.0 -cl-kernel-arg-info -cl-unsafe-math-optimizations -cl-single-precision-constant",
                             nullptr, nullptr);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Unable to build OpenCL program");
        }

        cl_uint n;
        err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &n, nullptr);

        if (n == 0 || err != CL_SUCCESS) {
            return false;
        }

        size_t sizes[n];
        err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, n * sizeof(size_t), &sizes[0], nullptr);

        if (err != CL_SUCCESS) {
            return false;
        }

        unsigned char *binaries[n];
        for (int i = 0; i < (int) n; ++i) {
            binaries[i] = new unsigned char[sizes[i]];
        }

        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, n * sizeof(unsigned char *), &binaries[0], nullptr);

        if (err != CL_SUCCESS) {
            return false;
        }

        auto cache_path = dehancer::device::get_opencl_cache_path();
        auto cache_file_name = get_cache_file_name(library_source);
        {
            std::ofstream ostrm(cache_path + "/" + cache_file_name, std::ios::binary);
            ostrm.write(reinterpret_cast<char *>(binaries[0]), sizes[0]);
        }

        for (int i = 0; i < (int) n; ++i) {
            delete binaries[i];
        }
        return true;
    }


}