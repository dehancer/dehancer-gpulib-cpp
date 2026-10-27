#include "../clut_identity.h"
#include "gtest/gtest.h"
#include "tests/include/run_test.h"
#include "tests/metal/paths_config.h"

TEST(TEST, CLUT_IDENTITY) {
	std::cout << std::endl;
	std::cerr << std::endl;

	make_identity("metal");
}
