//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"
#include "dehancer/gpu/DeviceCache.h"
#include "src/platforms/cuda/Utils.h"
#include "dehancer/gpu/kernels/cuda/texture.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename surfT>
cudaSurfaceObject_t createSurfaceObject(size_t width, size_t height, cudaArray* &opCuArray)
{
  cudaSurfaceObject_t surfObj = 0;

  assert(width > 0 && height > 0);

  //--- Allocate CUDA array in device memory ---
  cudaChannelFormatDesc channelDesc =
          cudaCreateChannelDesc<surfT>();

  CHECK_CUDA(
          cudaMallocArray(&opCuArray, &channelDesc, width, height,
                          cudaArraySurfaceLoadStore));

  //--- Specify sruface ---
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = opCuArray;

  // Create texture object
  CHECK_CUDA(cudaCreateSurfaceObject(&surfObj, &resDesc));

  return surfObj;
}

template< typename memT, typename imgT >
cv::Mat downloadCudaArrayToImage(cudaArray* iCuArray, size_t width, size_t height, CUstream stream)
{
  assert(sizeof(memT) >= sizeof (imgT));
  assert(width > 0 && height > 0);
  assert(iCuArray != nullptr);

  auto image = cv::Mat(
          height,
          width,
          CV_32FC4
  );

  //--- downlaod data ---

  CHECK_CUDA(cudaMemcpy2DFromArrayAsync(image.data,
                                        image.cols * sizeof(memT),
                                        iCuArray,
                                        0, 0, width * sizeof(memT),  height,
                                        cudaMemcpyDeviceToHost, stream));

  return image;
}

TEST(TEST, DeviceCache_OpenCL) {

  size_t width = 400*4, height = 300*4;
  size_t levels = 16;

  std::cout << std::endl;
  std::cerr << std::endl;

  auto* command_queue = dehancer::DeviceCache::Instance().get_default_command_queue();

  // Create module from binary file
  CUmodule cuModule;
  std::cout << "Module: " << dehancer::device::get_lib_path() << std::endl;
  CHECK_CUDA(cuModuleLoad(&cuModule, dehancer::device::get_lib_path().c_str()));

  // Get function handle from module
  CUfunction kernel_surface_gen;
  CHECK_CUDA(cuModuleGetFunction(&kernel_surface_gen, cuModule, "kernel_grid"));


  cudaArray* cuArray;
  cudaSurfaceObject_t target = createSurfaceObject<float4>(width, height, cuArray);;


  // Invoke kernel
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y,
               1
  );

  dehancer::nvcc::texture2d<float4> target_text {
    .surface = target,
    .width = width,
    .height = height
  };

  std::vector<void*> args; args.resize(2);
  args[0] = &levels;
  args[1] = &target_text;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  printf("Launching CUDA Kernel\n");

  //--- Record the start event ---
  CHECK_CUDA(cudaEventRecord(start, nullptr));

  CHECK_CUDA(cuLaunchKernel(
          kernel_surface_gen,
          dimGrid.x, dimGrid.y, dimGrid.z,
          dimBlock.x, dimBlock.y, dimBlock.z,
          0,
          static_cast<CUstream>(command_queue),
          args.data(),
          nullptr)
  );

  //--- Record the stop event ---
  CHECK_CUDA(cudaEventRecord(stop, nullptr));

  //--- Wait for the stop event to complete ---
  CHECK_CUDA(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

  cv::Mat cv_result = downloadCudaArrayToImage<float4, float4>(cuArray, width, height, static_cast<CUstream>(command_queue));

  auto output_type = CV_16U;
  auto output_color = cv::COLOR_RGBA2BGR;
  auto scale = 256.0f*256.0f;

  cv_result.convertTo(cv_result, output_type, scale);
  cv::cvtColor(cv_result, cv_result, output_color);

  cv::imwrite("texture.png", cv_result);

  printf("Input Size  [%dx%d], ", cv_result.size[0], cv_result.size[1]);
  printf("GPU processing time : %.4f (ms)\n", msecTotal);
  printf("Pixel throughput    : %.3f Mpixels/sec\n",
         ((float)(width * height*1000.f)/msecTotal)/1000000);
  printf("------------------------------------------------------------------\n");


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