//
// Created by denn nevera on 15/11/2020.
//

#include "../function.h"
#include "gtest/gtest.h"
#include "tests/include/run_test.h"

TEST(TEST, OPENCL_GAUSSIAN_FUNCTUON) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("opencl", gaussian_test);
}

TEST(TEST, OPENCL_BOX_BLUR_FUNCTUON) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("opencl", box_test);
}
