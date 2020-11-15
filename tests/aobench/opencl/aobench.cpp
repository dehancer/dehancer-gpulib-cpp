//
// Created by denn nevera on 09/11/2020.
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
  float       compression = 0.0f;

  size_t width = 800, height = 600;

  auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));

  auto bench_kernel = dehancer::Function(command_queue, "ao_bench_kernel", true);
  auto ao_bench_text = bench_kernel.make_texture(width,height);

  std::chrono::time_point<std::chrono::system_clock> clock_begin
          = std::chrono::system_clock::now();
  /***
   * Test performance
   */
  bench_kernel.execute([&ao_bench_text](dehancer::CommandEncoder& command_encoder){
      int numSubSamples = 4, count = 0;

      command_encoder.set(&numSubSamples, sizeof(numSubSamples), count++);
      command_encoder.set(ao_bench_text, count++);

      return ao_bench_text;
  });


  std::chrono::time_point<std::chrono::system_clock> clock_end
          = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = clock_end-clock_begin;

  // Report results and save image
  std::cout << "[aobench cl ("<<dehancer::device::get_name(device)<<")]:\t" << seconds.count() << "s "
            << ", for a " << width << "x" << height << " pixels" << std::endl;


  std::string out_file_cv = "ao-cl-"; out_file_cv.append(std::to_string(num)); out_file_cv.append(ext);

  {
    std::ofstream ao_bench_os(out_file_cv, std::ostream::binary | std::ostream::trunc);
    ao_bench_os << dehancer::TextureOutput(command_queue, ao_bench_text, {
            .type = type,
            .compression = compression
    });
  }

  /***
   * Test blend and write output
   */
  auto blend_kernel = dehancer::Function(command_queue, "blend_kernel");
  auto input_text = dehancer::TextureInput(command_queue);

  std::ifstream ifs(out_file_cv, std::ios::binary);
  ifs >> input_text;
  auto source = input_text.get_texture();
  auto result = blend_kernel.make_texture(width,height);

  blend_kernel.execute([&source, &result](dehancer::CommandEncoder& command_encoder){
      int count = 0;

      command_encoder.set(source, count++);
      command_encoder.set(result, count++);

      return result;
  });

  std::string out_file_result = "ao-cl-result-"; out_file_result.append(std::to_string(num)); out_file_result.append(ext);
  {
    std::ofstream result_os(out_file_result, std::ostream::binary | std::ostream::trunc);
    result_os << dehancer::TextureOutput(command_queue, result, {
            .type = type,
            .compression = compression
    });
  }

  dehancer::DeviceCache::Instance().return_command_queue(command_queue);

  return 0;
}

TEST(TEST, AOBENCH_OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  try {
    auto devices = dehancer::DeviceCache::Instance().get_device_list();
    assert(!devices.empty());

    void* device;

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