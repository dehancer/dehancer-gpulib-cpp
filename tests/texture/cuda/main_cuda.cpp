//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../io_texture.h"
#include "../../include/run_images.h"
#include "tests/cuda/paths_config.h"


TEST(TEST, CUDA_TEXTURE) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_images("cuda", io_texture_test);

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      return CUDA_KERNELS_LIBRARY;// + std::string("++");
    }
}