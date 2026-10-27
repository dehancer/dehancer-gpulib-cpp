#include "../function.h"
#include "gtest/gtest.h"
#include "tests/include/run_test.h"

TEST(TEST, OPENCL_RESAMPLER) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("opencl", function_test);
}
