#include "../convolve_test.h"
#include "gtest/gtest.h"
#include "tests/cuda/paths_config.h"

TEST(TEST, CUDA_CONVOLVE) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run("cuda");
}
