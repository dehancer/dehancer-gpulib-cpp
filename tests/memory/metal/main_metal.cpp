//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../memory_test.h"
#include "../../include/run_test.h"
#include "tests/metal/paths_config.h"


TEST(TEST, METAL_FUNCTUON) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_on_devices("metal", memory_test);

}
