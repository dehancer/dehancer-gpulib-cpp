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

int run_bench2(int num, const void* device) {

  dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
  std::string ext = dehancer::TextureIO::extention_for(type);
  float compression = 0.0f;

  size_t width = 800, height = 600;

  auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));

  auto bench_kernel = dehancer::Function(command_queue, "ao_bench_kernel", true);
  auto ao_bench_text = bench_kernel.make_texture(width, height);

  std::chrono::time_point<std::chrono::system_clock> clock_begin
          = std::chrono::system_clock::now();
  /***
   * Test performance
   */
//  bench_kernel.execute([&ao_bench_text](dehancer::CommandEncoder& command_encoder){
//      int numSubSamples = 4, count = 0;
//
//      command_encoder.set(&numSubSamples, sizeof(numSubSamples), count++);
//      command_encoder.set(ao_bench_text, count++);
//
//      return ao_bench_text;
//  });

  return 0;

}

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

    for (auto d: devices) {
      if (run_bench2(dev_num++, d)!=0) return;
    }

    //if (run_bench2(0, devices[0])!=0) return;

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}