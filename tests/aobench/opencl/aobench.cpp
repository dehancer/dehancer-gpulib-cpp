#include "tests/aobench/aobench.h"

#include "gtest/gtest.h"

TEST(TEST, AOBENCH_OpenCL) {
	std::cout << std::endl;
	std::cerr << std::endl;

	test_bench("opencl");
}
