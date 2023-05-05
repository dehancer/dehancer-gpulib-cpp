//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../function.h"
#include "tests/include/run_test.h"
#include "tests/metal/paths_config.h"

TEST(TEST, METAL_OPTICAL_RESOLUTION) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_images("metal", function_test);

}
