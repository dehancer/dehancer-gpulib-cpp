//
// Created by denn nevera on 29/11/2020.
//


#include "gtest/gtest.h"
#include "../blur.h"

TEST(TEST, AOBENCH_OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  test_bench("opencl");

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      static std::string path = "blurKernel.cl";
      return path;
    }
}