#include "../cache.h"
#include "../load.h"
#include "gtest/gtest.h"
#include "tests/include/run_test.h"

TEST(TEST, LOAD_CUBE) {
	std::cout << std::endl;
	std::cerr << std::endl;

	load_from_cube("opencl");
}

TEST(TEST, LOAD_CACHED_ACES) {
	std::cout << std::endl;
	std::cerr << std::endl;

	load_from_cache("opencl");
}
