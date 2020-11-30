//
// Created by denn nevera on 09/11/2020.
//

#include "gtest/gtest.h"
#include "tests/aobench/aobench.h"

TEST(TEST, AOBENCH_OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  test_bench("opencl");

}
