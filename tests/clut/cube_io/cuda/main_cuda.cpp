#include "../cache.h"
#include "../load.h"
#include "gtest/gtest.h"
#include "tests/cuda/paths_config.h"
#include "tests/include/run_test.h"

TEST(TEST, LOAD_CUBE) {
	std::cout << std::endl;
	std::cerr << std::endl;

	load_from_cube("cuda");
}

TEST(TEST, LOAD_CACHED_ACES) {
	std::cout << std::endl;
	std::cerr << std::endl;

	load_from_cache("cuda");
}
