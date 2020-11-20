//
// Created by denn nevera on 16/11/2020.
//

#pragma once

#include "dehancer/gpu/Lib.h"

#include <chrono>

namespace test {

    using namespace dehancer;

    std::vector<float3> false_color_map = {
            {0.28,0.16,0.31}, // 0
            {0.16,0.36,0.53}, // 1
            {0.22,0.48,0.61}, // 2
            {0.30,0.59,0.65}, // 3
            {0.42,0.60,0.63}, // 4
            {0.51,0.55,0.48}, // 5
            {0.48,0.70,0.35}, // 6
            {0.59,0.64,0.59}, // 7
            {0.84,0.53,0.50}, // 8
            {0.87,0.66,0.64}, // 9
            {0.85,0.86,0.78}, // 10
            {0.95,0.93,0.52}, // 11
            {1.00,0.97,0.33}, // 12
            {0.95,0.65,0.23}, // 13
            {0.93,0.27,0.18}, // 14
            {0.80,0.10,0.05}  // 15
    };

    std::vector<float> get_as_mem(std::vector<float3> map) {
      std::vector<float> list;
      for(auto v: map) {
        for (auto p: v) {
          list.push_back(p);
        }
      }
      return list;
    }

    using namespace dehancer;
    class BlendKernel: public Kernel {
    public:
        BlendKernel(const void* command_queue, const Texture& s, const Texture& d):
        dehancer::Kernel(command_queue,"blend_kernel", s, d),
        color_map_(nullptr),
        opacity_{0.1,0.5,0.5}
        {
          auto map_data = get_as_mem(false_color_map);
          color_map_ = MemoryHolder::Make(get_command_queue(),
                                          map_data.data(),
                                          map_data.size()*sizeof(float));
          levels_ = static_cast<uint>(false_color_map.size());
        };

        void setup(CommandEncoder &encode) override {
          encode.set(color_map_,2);
          encode.set(&levels_,sizeof(levels_),3);
          encode.set(opacity_,4);
        }

    private:
        Memory color_map_;
        uint   levels_;
        float3 opacity_;
    };
}

int run_bench2(int num, const void* device, std::string patform) {

  dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
  std::string ext = dehancer::TextureIO::extention_for(type);
  float       compression = 0.3f;

  size_t width = 800*2, height = 600*2;

  auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));

  auto bench_kernel = dehancer::Function(command_queue, "ao_bench_kernel", true);
  auto ao_bench_text = bench_kernel.make_texture(width,height);

  /**
   * Debug info
   */

  std::cout << "[aobench kernel " << bench_kernel.get_name() << " args: " << std::endl;
  for (auto& a: bench_kernel.get_arg_list()) {
    std::cout << std::setw(20) << a.name << "["<<a.index<<"]: " << a.type_name << std::endl;
  }


  std::chrono::time_point<std::chrono::system_clock> clock_begin
          = std::chrono::system_clock::now();
  /***
   * Test performance
   */
  bench_kernel.execute([&ao_bench_text](dehancer::CommandEncoder& command_encoder){
      int numSubSamples = 4;

      command_encoder.set(&numSubSamples, sizeof(numSubSamples), 0);
      command_encoder.set(ao_bench_text, 1);

      return ao_bench_text;
  });


  std::chrono::time_point<std::chrono::system_clock> clock_end
          = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = clock_end-clock_begin;

  // Report results and save image
  auto device_type = dehancer::device::get_type(device);

  std::string device_type_str;

  switch (device_type) {
    case dehancer::device::Type::cpu :
      device_type_str = "CPU"; break;
    case dehancer::device::Type::gpu :
      device_type_str = "GPU"; break;
    default:
      device_type_str = "Unknown"; break;
  }

  std::cout << "[aobench "
            <<patform<<"/"<<device_type_str
            <<" ("
            <<dehancer::device::get_name(device)
            <<")]:\t" << seconds.count() << "s "
            << ", for a " << width << "x" << height << " pixels" << std::endl;


  std::string out_file_cv = "ao-"+patform+"-"; out_file_cv.append(std::to_string(num)); out_file_cv.append(ext);

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
  auto input_text = dehancer::TextureInput(command_queue);
  auto output_text = dehancer::TextureOutput(command_queue, width, height, nullptr, {
          .type = type,
          .compression = compression
  });

  std::ifstream ifs(out_file_cv, std::ios::binary);
  ifs >> input_text;

  auto blend_kernel = test::BlendKernel(
          command_queue,
          input_text.get_texture(),
          output_text.get_texture());

  blend_kernel.process();

  std::string out_file_result = "ao-"+patform+"-result-"; out_file_result.append(std::to_string(num)); out_file_result.append(ext);
  {
    std::ofstream result_os(out_file_result, std::ostream::binary | std::ostream::trunc);
    result_os << output_text;
  }

  dehancer::DeviceCache::Instance().return_command_queue(command_queue);

  return 0;
}

void test_bench(std::string platform) {
  try {
    auto devices = dehancer::DeviceCache::Instance().get_device_list();
    assert(!devices.empty());

    int dev_num = 0;
    std::cout << "Platform: " << platform << std::endl;
    // for (auto d: devices) {
    // std::cout << " #" << dev_num++ << std::endl;
    // std::cout << "    Device '" << dehancer::device::get_name(d) << " ["<<dehancer::device::get_id(d)<<"]'"<< std::endl;
    // }

    // std::cout << "Bench: " << std::endl;
    // dev_num = 0;

    for (auto d: devices) {
#if __APPLE__
      if (dehancer::device::get_type(d) == dehancer::device::Type::cpu) continue;
#endif
      if (run_bench2(dev_num++, d, platform)!=0) return;
    }

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}