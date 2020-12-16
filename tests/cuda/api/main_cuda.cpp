//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "dehancer/gpu/DeviceCache.h"
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

//static const char *_cudaGetErrorEnum(cudaError_t error) {
//  return cudaGetErrorName(error);
//}

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

  // Create context
  CUcontext cuContext;
  (cuCtxCreate(&cuContext, 0, cuDevice));

  cudaStream_t s0;
  checkCudaErrors(cudaStreamCreate(&s0));

  // Check stream
  CUcontext cuContext_0;
  checkCudaErrors(cuStreamGetCtx(s0, &cuContext_0));

  checkCudaErrors(cuCtxPopCurrent(&cuContext));

  checkCudaErrors(cuCtxPushCurrent(cuContext_0));

  CUdevice cUdevice_0 = -1;
  checkCudaErrors(cuCtxGetDevice(&cUdevice_0));

  cudaDeviceProp props{};

  cudaGetDeviceProperties(&props, cUdevice_0);

  std::cout << ">> checking ...." << std::endl;
  print_device_info(cUdevice_0);


  checkCudaErrors(cuCtxDestroy(cuContext));

}