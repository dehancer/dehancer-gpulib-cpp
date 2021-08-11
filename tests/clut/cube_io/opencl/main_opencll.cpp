//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "tests/include/run_test.h"

#include "../load.h"
#include "../cache.h"

TEST(TEST, LOAD_CUBE) {

  std::cout << std::endl;
  std::cerr << std::endl;

  load_from_cube("opencl");

}

TEST(TEST, LOAD_CACHED_ACES) {

  std::cout << std::endl;
  std::cerr << std::endl;

  load_from_cache("opencl");

}
