//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../device_cache.h"
#include "tests/test_config.h"


TEST(TEST, DeviceCache_Cuda) {

  std::cout << std::endl;
  std::cerr << "Cuda device cache..." << std::endl;

  test_device();

}
