#include "../../include/run_test.h"
#include "../io_texture.h"
#include "gtest/gtest.h"
#include "tests/cuda/paths_config.h"

TEST(TEST, CUDA_TEXTURE) {
	std::cout << std::endl;
	std::cerr << std::endl;

	run_images("cuda", io_texture_test);
}
