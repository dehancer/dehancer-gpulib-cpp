//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "../io_texture.h"
#include "../../include/run_test.h"
#include "tests/metal/paths_config.h"


TEST(TEST, METAL_VIDEO_PLAYFORWARD) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_on_devices("metal_forward",io_texture_test_forward);

}

TEST(TEST, METAL_VIDEO_PLAYREVERSE) {

  std::cout << std::endl;
  std::cerr << std::endl;

  run_on_devices("metal_reverse",io_texture_test_reverse);

}

TEST(TEST, METAL_VIDEO_PLAY_LAST) {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  run_on_devices("metal_last",io_texture_test_last);
  
}
