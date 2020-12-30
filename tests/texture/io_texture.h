//
// Created by denn on 31.12.2020.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
[[maybe_unused]] std::string ext = dehancer::TextureIO::extention_for(type);
float       compression = 0.3f;

void io_texture_test(const std::string& path, const std::string& platform, void *command_queue, int im_num, int dev_num) {

  std::cout << "Load file: " << path << std::endl;

  auto input_text = dehancer::TextureInput(command_queue);

  std::ifstream ifs(path, std::ios::binary);
  ifs >> input_text;

  std::string out_file_cv = "texture-io-";
  out_file_cv.append(platform);
  out_file_cv.append("-["); out_file_cv.append(std::to_string(dev_num)); out_file_cv.append("]-");
  out_file_cv.append(std::to_string(im_num)); out_file_cv.append(ext);

  {
    std::ofstream os(out_file_cv, std::ostream::binary | std::ostream::trunc);
    if (os.is_open()) {
      os << dehancer::TextureOutput(command_queue, input_text.get_texture(), {
              .type = type,
              .compression = compression
      }) << std::flush;

      std::cout << "Save to: " << out_file_cv << std::endl;

    } else {
      std::cerr << "File: " << out_file_cv << " could not been opened..." << std::endl;
    }
  }
}

int run_bench(int num, const void* device, std::string patform) {
  auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));

  int i = 0;
  for (auto& file: IMAGE_FILES) {
    std::string path = IMAGES_DIR; path.append("/"); path.append(file);
    io_texture_test(path, patform, command_queue, i++, num);
  }

  dehancer::DeviceCache::Instance().return_command_queue(command_queue);
  return 0;
}

void test_bench(std::string platform) {
  try {
#if __APPLE__
    auto devices = dehancer::DeviceCache::Instance().get_device_list(
            dehancer::device::Type::gpu
            );
#else
    auto devices = dehancer::DeviceCache::Instance().get_device_list(
            dehancer::device::Type::gpu
    );
#endif
    assert(!devices.empty());

    int dev_num = 0;
    std::cout << "Platform: " << platform << std::endl;
    for (auto d: devices) {
      std::cout << " #" << dev_num++ << std::endl;
      std::cout << "    Device '" << dehancer::device::get_name(d) << " ["<<dehancer::device::get_id(d)<<"]'"<< std::endl;
    }

    std::cout << "Bench: " << std::endl;
    dev_num = 0;

    for (auto d: devices) {
      if (run_bench(dev_num++, d, platform)!=0) return;
    }

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}