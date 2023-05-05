//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"

#include "tests/cuda/paths_config.h"
#include "tests/filter/filter_test.h"
#include "tests/test_config.h"

TEST(TEST, FILTER_BASE_OpenCL) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_images("cuda", filter_test);
  
}
