//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"

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

void print_device_info (CUdevice device) {

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, device);

  std::cout << "CUDA device id: " << device<< std::endl;
  std::cout << "CUDA device uid: " ;
  for(int i=0; i< sizeof(props.uuid.bytes); ++i)
    std::cout << std::hex << (int)props.uuid.bytes[i];
  std::cout << std::endl;
  std::cout << "CUDA device multiProcessorCount: " << props.multiProcessorCount << std::endl;
  std::cout << "CUDA device maxThreadsPerBlock: " << props.maxThreadsPerBlock << std::endl;
  std::cout << "CUDA device maxThreadsDim: " << props.maxThreadsDim[0] << "x" << props.maxThreadsDim[1] << "x" << props.maxThreadsDim[2] << std::endl;
}

TEST(TEST, DeviceCache_OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  cuInit(0);

  int numDevices = 0;
  int maxMultiprocessors = 0, maxDevice = 0;

  cudaGetDeviceCount(&numDevices);

  std::cout << "CUDA device number: " << numDevices << std::endl;


  if (numDevices >= 1) {
    for (int device=0; device<numDevices; device++) {
      cudaDeviceProp props{};
      cudaGetDeviceProperties(&props, device);

      print_device_info(device);

      if (maxMultiprocessors < props.multiProcessorCount) {
        maxMultiprocessors = props.multiProcessorCount;
        maxDevice = device;
      }
    }
    cudaSetDevice(maxDevice);
  }

  // Get handle for device 0
  CUdevice cuDevice;
  checkCudaErrors(cuDeviceGet(&cuDevice, maxDevice));

  //cudaSetDevice(maxDevice);

  // Create context
  CUcontext cuContext;
  (cuCtxCreate(&cuContext, 0, cuDevice));

  cudaStream_t stream_0;
  checkCudaErrors(cudaStreamCreate(&stream_0));

  // Check stream
  CUcontext cuContext_0;
  checkCudaErrors(cuStreamGetCtx(stream_0, &cuContext_0));

  checkCudaErrors(cuCtxPopCurrent(&cuContext));

  checkCudaErrors(cuCtxPushCurrent(cuContext_0));

  CUdevice cUdevice_0 = -1;
  checkCudaErrors(cuCtxGetDevice(&cUdevice_0));

  cudaDeviceProp props{};

  cudaGetDeviceProperties(&props, cUdevice_0);

  std::cout << ">> checking ...." << std::endl;
  print_device_info(cUdevice_0);


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
    h_A[i] = i/2;
    h_B[i] = i%2;
  }

  // Allocate vectors in device memory
  //CUdeviceptr d_A; //cuMemAlloc(&d_A, size);
  uint8_t* d_A;
  checkCudaErrors(cudaMalloc(&d_A, size));

  CUdeviceptr d_B; cuMemAlloc(&d_B, size);

  //CUdeviceptr d_C; cuMemAlloc(&d_C, size);
  float* d_C;
  checkCudaErrors(cudaMalloc(&d_C, size));

  // Copy vectors from host memory to device memory
  //cuMemcpyHtoDAsync(d_A, h_A, size, stream_0);
  checkCudaErrors(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream_0));

  cuMemcpyHtoDAsync(d_B, h_B, size, stream_0);

  // Invoke kernel
  int threadsPerBlock = 64;
  int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

  void* args[] = { &d_A, &d_B, &d_C, &N };

  checkCudaErrors(cuLaunchKernel(
          vecAdd,
          blocksPerGrid, 1, 1,
          threadsPerBlock, 1, 1,
          0,
          stream_0,
          args,
          nullptr)
  );

  //cuMemcpyDtoHAsync(h_A, d_C, size, stream_0);
  memset(h_A,0,size);
  checkCudaErrors(cudaMemcpyAsync(h_A, d_C, size, cudaMemcpyDeviceToHost, stream_0));

  std::cout << "summ: " << std::endl;
  for (int i = 0; i < N; ++i) {
    std::cout << h_A[i] << " ";
  }
  std::cout << std::endl;

  free(h_A);
  free(h_B);

  cudaFree(d_A);
  cuMemFree(d_B);
  cudaFree(d_C);

  checkCudaErrors(cuCtxDestroy(cuContext));

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