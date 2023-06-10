//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
//#include "tests/paths_config.h"

#include "tests/aobench/aobench.h"

TEST(TEST, AOBENCH_Metal) {

  std::cout << std::endl;
  std::cerr << std::endl;

  test_bench("metal");

}
