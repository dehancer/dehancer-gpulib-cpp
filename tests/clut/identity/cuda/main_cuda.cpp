//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../clut_identity.h"
#include "tests/include/run_test.h"
#include "tests/cuda/paths_config.h"

TEST(TEST, CLUT_IDENTITY) {

  std::cout << std::endl;
  std::cerr << std::endl;

  make_identity("cuda");

}

