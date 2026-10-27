//
// Created by denn nevera on 15/11/2020.
//

#include "../clut_transform.h"
#include "gtest/gtest.h"
#include "tests/cuda/paths_config.h"
#include "tests/include/run_test.h"

TEST(TEST, CLUT_IDENTITY) {
	std::cout << std::endl;
	std::cerr << std::endl;

	clut_transform("cuda");
}
