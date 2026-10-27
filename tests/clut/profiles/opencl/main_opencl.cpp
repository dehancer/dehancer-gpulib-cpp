//
// Created by denn nevera on 15/11/2020.
//

#include "../load.h"
#include "gtest/gtest.h"
#include "tests/include/run_test.h"

TEST(TEST, LOAD_XMP) {
	std::cout << std::endl;
	std::cerr << std::endl;

	load_from_xmp("opencl");
}
