//
// Created by denn on 01.01.2021.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

using dh_test_function = std::function<int (int num,
                                            const void* command_queue,
                                            const std::string& platform,
                                            const std::string& input_image,
                                            const std::string& output_image,
                                            int image_index)>;

using dh_test_on_devices_function = std::function<int (int num,
                                                       const void* command_queue,
                                                       const std::string& platform)>;

using dh_test_on_grid_image_function = std::function<int (int num,
                                                          const void* command_queue,
                                                          const dehancer::Texture& texture,
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
      
      auto ret = block(dev_num++, command_queue, platform) != 0;
      
      dehancer::DeviceCache::Instance().return_command_queue(command_queue);
      
      if (ret) return;
    }
    
  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}


inline static void run_on_grid_image(std::string platform, dh_test_on_grid_image_function block) {
  
  auto test_on_grid_image =  [block] (int dev_num,
                                 const void* command_queue,
                                 const std::string& platform) {
      
      dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
      std::string ext = dehancer::TextureIO::extention_for(type);
      float compression = 0.3f;
      
      size_t width = 800*2, height = 400*2 ;
      
      auto grid_kernel = dehancer::Function(command_queue,"kernel_grid");
      auto grid_text = grid_kernel.make_texture(width, height);
      
      /***
    * Test performance
    */
      grid_kernel.execute([&grid_text](dehancer::CommandEncoder& command_encoder){
          int levels = 6;
          
          command_encoder.set(levels, 0);
          command_encoder.set(grid_text, 1);
          
          return dehancer::CommandEncoder::Size::From(grid_text);
      });
      
      std::string out_file_cv = "grid-"+platform+"-"; out_file_cv.append(std::to_string(dev_num)); out_file_cv.append(ext);
      
      {
        std::ofstream ao_bench_os(out_file_cv, std::ostream::binary | std::ostream::trunc);
        ao_bench_os << dehancer::TextureOutput(command_queue, grid_text, {
                .type = type,
                .compression = compression
        });
      }
      
      return block(dev_num, command_queue, grid_text, platform);
  };
  
  run_on_devices(platform, test_on_grid_image);
}