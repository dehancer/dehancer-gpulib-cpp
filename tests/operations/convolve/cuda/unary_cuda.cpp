//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"

#include "tests/cuda/paths_config.h"
#include "../convolve_test.h"

TEST(TEST, CUDA_CONVOLVE) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run("cuda");
  
}
