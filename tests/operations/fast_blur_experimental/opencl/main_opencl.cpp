//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../fast_blur_test.h"
#include "tests//include/run_test.h"


TEST(TEST, OPENCL_FAST_BLUR) {

  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_on_grid_image("opencl", fast_blur_test);

}

//TEST(TEST, OPENCL_GAUSSIAN_BOXED) {
//
//  std::cout << std::endl;
//  std::cerr << std::endl;
//
//  run_on_grid_image("opencl", gaussian_boxed_blur_test);
//
//}