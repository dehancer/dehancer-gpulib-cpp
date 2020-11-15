//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"

#include "dehancer/gpu/DeviceCache.h"

#include <chrono>

TEST(TEST, DeviceCache_OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  auto* command_queue = dehancer::DeviceCache::Instance().get_default_command_queue();

  for (int i = 0; i < 100 ; ++i) {
    std::vector<void*> queues;
    for (int j = 0; j < 32; ++j) {
      auto* q = dehancer::DeviceCache::Instance().get_default_command_queue();
      queues.push_back(q);
    }
    for (auto q: queues){
      dehancer::DeviceCache::Instance().return_command_queue(q);
    }
  }

  dehancer::DeviceCache::Instance().return_command_queue(command_queue);

}