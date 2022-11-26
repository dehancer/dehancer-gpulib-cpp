//
// Created by denn nevera on 15/11/2020.
//

//#import <Metal/Metal.h>

#include "gtest/gtest.h"

#include "gtest/gtest.h"
#include "dehancer/gpu/DeviceCache.h"
#include <chrono>

#include "../device_cache.h"

TEST(TEST, DeviceCache_Metal) {

  std::cout << std::endl;
  std::cerr << "Metal device cache..." << std::endl;

  test_device();

}