//
// Created by denn nevera on 15/11/2020.
//

#include "../../include/run_test.h"
#include "../io_texture.h"
#include "gtest/gtest.h"
#include "tests/metal/paths_config.h"

TEST(TEST, CUDA_TEXTURE) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("metal", io_texture_test);
}
