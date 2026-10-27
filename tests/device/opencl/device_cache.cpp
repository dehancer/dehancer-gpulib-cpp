#include "../device_cache.h"

#include "gtest/gtest.h"

TEST(TEST, DeviceCache_OpenCL) {
	std::cout << std::endl;
	std::cerr << "OpenCL device cache..." << std::endl;

	test_device();
}
