//
// Created by denn nevera on 29/11/2020.
//


#include "gtest/gtest.h"
#include "tests/filter/filter_test.h"

TEST(TEST, FILTER_BASE_OpenCL) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_images("opencl", filter_test);
  
}
