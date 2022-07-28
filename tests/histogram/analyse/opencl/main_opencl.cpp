//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../function.h"
#include "tests/include/run_test.h"

TEST(TEST, OPENCL_CLEAR_HISTOGRAM) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_images("opencl", function_test, IMAGE_CLEAR_FILES);

}

TEST(TEST, OPENCL_REAL_HISTOGRAM) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_images("opencl", function_test, IMAGE_ONE_FILE);

}
