//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"

#include "tests/metal/paths_config.h"
#include "tests/filter/filter_test.h"

TEST(TEST, FILTER_BASE_METAL) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_images("metal", filter_test);
  
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