//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../load.h"
#include "tests/include/run_test.h"
#include "tests/metal/paths_config.h"

TEST(TEST, LOAD_XMP) {

  std::cout << std::endl;
  std::cerr << std::endl;

  load_from_xmp("metal");

}
