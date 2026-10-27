#include "../function.h"
#include "gtest/gtest.h"
#include "tests/include/run_test.h"
#include "tests/metal/paths_config.h"

TEST(TEST, METAL_MORPH) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("metal", function_test);
}
