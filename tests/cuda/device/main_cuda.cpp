//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"

#include "dehancer/gpu/DeviceCache.h"

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(static_cast<cudaError_t>(result)), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


TEST(TEST, DeviceCache_OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  auto* command_queue = dehancer::DeviceCache::Instance().get_default_command_queue();

  // Create module from binary file
  CUmodule cuModule;
  std::cout << "Module: " << dehancer::device::get_lib_path() << std::endl;
  checkCudaErrors(cuModuleLoad(&cuModule, dehancer::device::get_lib_path().c_str()));

  // Get function handle from module
  CUfunction vecAdd;
  checkCudaErrors(cuModuleGetFunction(&vecAdd, cuModule, "kernel_vec_add"));


  int N = 1024;
  size_t size = N * sizeof(float);

  // Allocate input vectors h_A and h_B in host memory
  auto* h_A = (float*)malloc(size);
  auto* h_B = (float*)malloc(size);

  // Initialize input vectors
  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = i%2;
  }

  // Allocate vectors in device memory
  CUdeviceptr d_A; checkCudaErrors(cuMemAlloc(&d_A, size));

  CUdeviceptr d_B; checkCudaErrors(cuMemAlloc(&d_B, size));

  CUdeviceptr d_C; checkCudaErrors(cuMemAlloc(&d_C, size));

  // Copy vectors from host memory to device memory
  //checkCudaErrors(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, static_cast<CUstream>(command_queue));
  checkCudaErrors(cuMemcpyHtoDAsync(d_A, h_A, size, static_cast<CUstream>(command_queue)));
  checkCudaErrors(cuMemcpyHtoDAsync(d_B, h_B, size, static_cast<CUstream>(command_queue)));

  // Invoke kernel
  int threadsPerBlock = 64;
  int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

  void* args[] = { &d_A, &d_B, &d_C, &N };

  checkCudaErrors(cuLaunchKernel(
          vecAdd,
          blocksPerGrid, 1, 1,
          threadsPerBlock, 1, 1,
          0,
          static_cast<CUstream>(command_queue),
          args,
          nullptr)
  );

  checkCudaErrors(cuMemcpyDtoHAsync(h_A, d_C, size, static_cast<CUstream>(command_queue)));

  std::cout << "summ: " << std::endl;
  for (int i = 0; i < N; ++i) {
    std::cout << h_A[i] << " ";
  }
  std::cout << std::endl;

  free(h_A);
  free(h_B);

  checkCudaErrors(cuMemFree(d_A));
  checkCudaErrors(cuMemFree(d_B));
  checkCudaErrors(cuMemFree(d_C));

  dehancer::DeviceCache::Instance().return_command_queue(command_queue);

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      return CUDA_KERNELS_LIBRARY;// + std::string("++");
    }
}