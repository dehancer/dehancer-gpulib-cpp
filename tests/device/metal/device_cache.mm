//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"

#include "gtest/gtest.h"
#include "dehancer/gpu/DeviceCache.h"
#include <chrono>

#include "../device_cache.h"
#include "tests/test_config.h"
#include "tests/metal/paths_config.h"

TEST(TEST, DeviceCache_Metal) {

  std::cout << std::endl;
  std::cerr << "Metal device cache..." << std::endl;

  test_device();

}
