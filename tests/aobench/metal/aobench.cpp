//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/TextureInput.h"
#include "dehancer/gpu/TextureOutput.h"
#include "dehancer/gpu/DeviceCache.h"

#include <chrono>

TEST(TEST, AOBENCH_METAL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  try {
    auto devices = dehancer::DeviceCache::Instance().get_device_list();
    assert(!devices.empty());

    int dev_num = 0;
    std::cout << "Info: " << std::endl;
    for (auto d: devices) {
      std::cout << " #" << dev_num++ << std::endl;
      std::cout << "    Device '" << dehancer::device::get_name(d) << " ["<<dehancer::device::get_id(d)<<"]'"<< std::endl;
    }

    std::cout << "Bench: " << std::endl;
    dev_num = 0;

//    for (auto d: devices) {
//      if (run_bench2(dev_num++, d)!=0) return;
//    }

    //if (run_bench2(0, devices[0])!=0) return;

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}