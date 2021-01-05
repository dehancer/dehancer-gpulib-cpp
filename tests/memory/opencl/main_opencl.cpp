//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../memory_test.h"
#include "../../include/run_images.h"
#include "tests/cuda/paths_config.h"


TEST(TEST, CUDA_FUNCTUON) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_on_devices("opencl", memory_test);
  
}