//
// Created by denn nevera on 15/11/2020.
//

#include "../function.h"
#include "gtest/gtest.h"
#include "tests/cuda/paths_config.h"
#include "tests/include/run_test.h"

TEST(TEST, CUDA_OPTICAL_RESOLUTION) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("cuda", function_test);
}
