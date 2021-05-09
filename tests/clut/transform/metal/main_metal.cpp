//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "tests/include/run_test.h"
#include "tests/metal/paths_config.h"
#include "../clut_transform.h"

#include <iostream>

TEST(TEST, CLUT_IDENTITY) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  clut_transform("metal");
  
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