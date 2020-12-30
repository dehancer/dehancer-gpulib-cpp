//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../io_texture.h"


TEST(TEST, OPENCL_IO_TEXTURE) {

  std::cout << std::endl;
  std::cerr << std::endl;

  test_bench("opencl");

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */

}