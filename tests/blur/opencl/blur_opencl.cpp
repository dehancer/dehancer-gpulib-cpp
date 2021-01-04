//
// Created by denn nevera on 29/11/2020.
//


#include "gtest/gtest.h"
#include "../blur_test.h"

TEST(TEST, AOBENCH_OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run("opencl");

}
