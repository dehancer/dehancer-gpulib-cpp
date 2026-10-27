#include "../space_transform.h"
#include "gtest/gtest.h"
#include "tests/cuda/paths_config.h"
#include "tests/include/run_test.h"

TEST(TEST, SPACE_TRANSFORM) {
	std::cout << std::endl;
	std::cerr << std::endl;

	load_from_cache("cuda");
}
