//
// Created by denn on 01.01.2021.
//

#pragma once

using dh_test_function = std::function<int (int num,
                                            const void* command_queue,
                                            const std::string& platform,
                                            const std::string& input_image,
                                            const std::string& output_image,
                                            int image_index)>;

using dh_test_on_devices_function = std::function<int (int num,
                                            const void* command_queue,
                                            const std::string& platform)>;


inline static int run_on_device(int num, const void* device, std::string platform, dh_test_function block) {
  auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));

  int i = 0;
  for (auto& file: IMAGE_FILES) {
    std::string path = IMAGES_DIR; path.append("/"); path.append(file);

    std::string out_file_cv = "texture-io-";
    out_file_cv.append(platform);
    out_file_cv.append("-["); out_file_cv.append(std::to_string(num)); out_file_cv.append("]-");
    out_file_cv.append(std::to_string(i)); out_file_cv.append(test::ext);

    auto r = block (num, command_queue, platform, path, out_file_cv, i++);
    if (r!=0) return r;
  }

  dehancer::DeviceCache::Instance().return_command_queue(command_queue);

  return 0;
}

inline static void run_images(std::string platform, dh_test_function block) {
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
      if (run_on_device(dev_num++, d, platform, block) != 0) return;
    }

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

inline static void run_on_devices(std::string platform, dh_test_on_devices_function block) {
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
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(d));
      if (block(dev_num++, command_queue, platform) != 0) {
        return;
      }
      dehancer::DeviceCache::Instance().return_command_queue(command_queue);
    }

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}