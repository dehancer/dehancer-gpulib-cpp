#include "../convolve_test.h"
#include "gtest/gtest.h"

TEST(TEST, OPENCL_CONVOLVE) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run("opencl");
}
