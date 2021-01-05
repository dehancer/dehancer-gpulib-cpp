//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../fast_blur_test.h"
#include "../../include/run_test.h"
#include "tests/cuda/paths_config.h"


TEST(TEST, CUDA_FUNCTUON) {

  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_on_grid_image("opencl", fast_blur_test);

}