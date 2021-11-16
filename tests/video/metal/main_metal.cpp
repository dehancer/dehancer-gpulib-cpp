//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../io_texture.h"
#include "../../include/run_test.h"
#include "tests/metal/paths_config.h"


TEST(TEST, METAL_VIDEO_PLAYFORWARD) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_on_devices("metal",io_texture_test_forward);

}

TEST(TEST, METAL_VIDEO_PLAYREVERSE) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_on_devices("metal",io_texture_test_reverse);
  
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