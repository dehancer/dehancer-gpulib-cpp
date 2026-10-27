#include "gtest/gtest.h"
#include "tests/aobench/aobench.h"
#include "tests/cuda/paths_config.h"

TEST(TEST, AOBENCH_Metal) {
	std::cout << std::endl;
	std::cerr << std::endl;

	test_bench("cuda");
}
