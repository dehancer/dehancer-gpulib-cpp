//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../io_texture.h"
#include "../../include/run_test.h"


TEST(TEST, OPENCL_IO_TEXTURE) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_images("opencl", io_texture_test);

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */

}