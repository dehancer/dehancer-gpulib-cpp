//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"
#include "dehancer/gpu/DeviceCache.h"
#include "src/platforms/cuda/utils.h"
#include "dehancer/gpu/kernels/cuda/texture1d.h"
#include "dehancer/gpu/kernels/cuda/texture2d.h"
#include "dehancer/gpu/kernels/cuda/texture3d.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

template< typename memT>
cv::Mat cudaArrayToImage(cudaArray* iCuArray, size_t width, size_t height, CUstream stream)
{
  assert(sizeof(memT) >= sizeof (::float4));
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

TEST(TEST, CUDA_TEXTURE_LOW_LAYER) {

  size_t width = 400*4, height = 300*4;
  size_t scaled_width = width*1.5, scaled_height = height*1.5;
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

  // Get function handle from module
  CUfunction kernel_grid_test_transform;
  CHECK_CUDA(cuModuleGetFunction(&kernel_grid_test_transform, cuModule, "kernel_grid_test_transform"));

  // Get function handle from module
  CUfunction kernel_make3DLut_transform;
  CHECK_CUDA(cuModuleGetFunction(&kernel_make3DLut_transform, cuModule, "kernel_make3DLut_transform"));

  // Get function handle from module
  CUfunction kernel_make1DLut_transform;
  CHECK_CUDA(cuModuleGetFunction(&kernel_make1DLut_transform, cuModule, "kernel_make1DLut_transform"));


  // Generated texture
  dehancer::nvcc::texture3d<::float4> clut(64,64,64);
  dehancer::nvcc::texture1d<::float4> clut_curve(256);

  dehancer::nvcc::texture2d<::float4> grid_target(width,height);

  dim3 dimBlockLut(8, 8, 8);
  dim3 dimGridLut((clut.get_width()  + dimBlockLut.x - 1) / dimBlockLut.x,
                  (clut.get_height() + dimBlockLut.y - 1) / dimBlockLut.y,
                  (clut.get_depth() + dimBlockLut.z - 1) / dimBlockLut.z
  );

  ::float2 compression_coeff = {1,0};
  std::vector<void*> lut_args = {  &clut, &compression_coeff };

  // Create identity LUT
  CHECK_CUDA(cuLaunchKernel(
          kernel_make3DLut_transform,
          dimGridLut.x, dimGridLut.y, dimGridLut.z,
          dimBlockLut.x, dimBlockLut.y, dimBlockLut.z,
          0,
          static_cast<CUstream>(command_queue),
          lut_args.data(),
          nullptr)
  );

  dim3 dimBlockCurve(16, 1, 1);
  dim3 dimGridCurve((clut_curve.get_width()  + dimBlockCurve.x - 1) / dimBlockCurve.x,
                  1,
                  1
  );

  ::float2 compression_coeff_curve = {0.2,0.5};
  std::vector<void*> curve_args = {  &clut_curve, &compression_coeff_curve };

  // Create identity Curve
  CHECK_CUDA(cuLaunchKernel(
          kernel_make1DLut_transform,
          dimGridCurve.x, dimGridCurve.y, dimGridCurve.z,
          dimBlockCurve.x, dimBlockCurve.y, dimBlockCurve.z,
          0,
          static_cast<CUstream>(command_queue),
          curve_args.data(),
          nullptr)
  );

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((grid_target.get_width()  + dimBlock.x - 1) / dimBlock.x,
               (grid_target.get_height() + dimBlock.y - 1) / dimBlock.y,
               1
  );

  dehancer::nvcc::texture2d<::float4> scaled_target(scaled_width,scaled_height);

  std::vector<void*> grid_args; grid_args.resize(2);
  grid_args[0] = &levels;
  grid_args[1] = &grid_target;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  printf("Launching CUDA Kernel\n");

  //--- Record the start event ---
  CHECK_CUDA(cudaEventRecord(start, nullptr));

  // Invoke kernels

  CHECK_CUDA(cuLaunchKernel(
          kernel_surface_gen,
          dimGrid.x, dimGrid.y, dimGrid.z,
          dimBlock.x, dimBlock.y, dimBlock.z,
          0,
          static_cast<CUstream>(command_queue),
          grid_args.data(),
          nullptr)
  );


  // yet another way dto initialize args
  std::vector<void*> output_args = {  &grid_target, &scaled_target, &clut, &clut_curve};

  dim3 dimGrid_scale((scaled_target.get_width()  + dimBlock.x - 1) / dimBlock.x,
                     (scaled_target.get_height() + dimBlock.y - 1) / dimBlock.y,
                     1
  );

  CHECK_CUDA(cuLaunchKernel(
          kernel_grid_test_transform,
          dimGrid_scale.x, dimGrid_scale.y, dimGrid_scale.z,
          dimBlock.x, dimBlock.y, dimBlock.z,
          0,
          static_cast<CUstream>(command_queue),
          output_args.data(),
          nullptr)
  );

  //--- Record the stop event ---
  CHECK_CUDA(cudaEventRecord(stop, nullptr));

  //--- Wait for the stop event to complete ---
  CHECK_CUDA(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&msecTotal, start, stop));

  cv::Mat cv_result = cudaArrayToImage<::float4>(scaled_target.get_contents(),
                                                 scaled_target.get_width(), scaled_target.get_height(),
                                                 static_cast<CUstream>(command_queue));

  auto output_type = CV_16U;
  auto output_color = cv::COLOR_RGBA2BGR;
  auto scale = 256.0f*256.0f;

  cv_result.convertTo(cv_result, output_type, scale);
  cv::cvtColor(cv_result, cv_result, output_color);

  cv::imwrite("texture.png", cv_result);

  printf("Input Size  [%dx%d], ", cv_result.size[0], cv_result.size[1]);
  printf("GPU processing time : %.4f (ms)\n", msecTotal);
  printf("Pixel throughput    : %.3f Mpixels/sec\n",
         ((float)(scaled_target.get_width() * scaled_target.get_height()*1000.f)/msecTotal)/1000000);
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