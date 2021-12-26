//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../io_texture.h"
#include "../../include/run_test.h"
#include "tests/cuda/paths_config.h"


TEST(TEST, CUDA_TEXTURE) {

  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_on_devices("cuda", io_texture_test_forward);

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      return CUDA_KERNELS_LIBRARY;
    }
    
    extern std::size_t get_lib_source(std::string& source) {
      return 0;
    }
}