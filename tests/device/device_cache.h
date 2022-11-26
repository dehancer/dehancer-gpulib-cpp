//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"

#include "dehancer/gpu/Lib.h"

#include <chrono>

inline static void test_device() {

  auto devices = dehancer::DeviceCache::Instance().get_device_list();

  for(auto d: devices) {
    auto name = dehancer::device::get_name(d);
    auto id = dehancer::device::get_id(d);
    auto type = dehancer::device::get_type(d);
    std::cout << "Cuda device["<<id<<"]: " << name << ": " << type << std::endl;
  }

  auto default_device = dehancer::DeviceCache::Instance().get_default_device();

  auto name = dehancer::device::get_name(default_device);
  auto id = dehancer::device::get_id(default_device);
  auto type = dehancer::device::get_type(default_device);

  std::cout << "Cuda default device["<<id<<"]: " << name << ": " << type << std::endl;

  auto* command_queue = dehancer::DeviceCache::Instance().get_default_command_queue();

  for (int i = 0; i < 100 ; ++i) {
    std::vector<void*> queues;
    for (int j = 0; j < 32; ++j) {
      auto* q = dehancer::DeviceCache::Instance().get_default_command_queue();
      
      if (!q) continue;
  
      dehancer::TextureDesc desc = {
              .width = 1920,
              .height = 1080
      };
  
      auto texture = desc.make(q);
  
      std::cout << "Metal Queue["<<static_cast<void*>(q)<<"]" << " texture length: " << texture->get_length() << std::endl;
  
      queues.push_back(q);
    }
    for (auto q: queues){
      dehancer::DeviceCache::Instance().return_command_queue(q);
    }
  }

  dehancer::DeviceCache::Instance().return_command_queue(command_queue);
}