//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../function.h"
#include "tests/include/run_test.h"
#include "tests/cuda/paths_config.h"
#include "tests/test_config.h"

TEST(TEST, CUDA_FUNCTUON) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_images("cuda", function_test);

}
