//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../function.h"
#include "tests/include/run_test.h"
#include "tests/cuda/paths_config.h"

TEST(TEST, CUDA_GAMMA) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_images("cuda", function_test);

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      return METAL_KERNELS_LIBRARY;
    }

    extern std::size_t get_lib_source(std::string& source) {
      return 0;
    }
}