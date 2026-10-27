//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "tests/filter/filter_test.h"
#include "tests/metal/paths_config.h"

TEST(TEST, FILTER_BASE_METAL) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("metal", filter_test);
}
