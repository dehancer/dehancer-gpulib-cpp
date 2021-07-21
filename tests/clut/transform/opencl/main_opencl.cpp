//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "tests/include/run_test.h"
#include "../clut_transform.h"

#include <iostream>

TEST(TEST, CLUT_IDENTITY) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  clut_transform("opencl");
  
}
