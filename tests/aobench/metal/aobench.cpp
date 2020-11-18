//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "tests/paths_config.h"

#include "../aobench.h"

TEST(TEST, AOBENCH_Metal) {

  std::cout << std::endl;
  std::cerr << std::endl;

  test_bench("metal");

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      return METAL_KERNELS_LIBRARY;
      //return "./TestKernels.metallib";
    }
}