//
// Created by denn on 24.04.2021.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "dehancer/MLutXmp.h"
#include "dehancer/Utils.h"
#include "tests/dotenv/dotenv_utils.h"
#include "tests/test_config.h"

void load_from_xmp(const std::string& platform) {
  
  std::cout << "Test " << std::endl;
  
  try {
  
    dotenv::dotenv::instance().config();
  
    auto pass = get_key();
  
    for (auto p: pass) std::cout<<(int)p<<",";
    std::cout<<std::endl;
  
    std::cout << std::endl;
  
    std::string file_path = DATA_DIR; file_path.append("/mlut.mlut");
    std::string cache_dir = "./cache/";
  
    dehancer::file::mkdir_p(cache_dir.c_str(),0777);
  
    std::cout << "Open test: " << file_path << std::endl;
  
    /*
     * * read properties
     * */
  
    auto xmp = dehancer::MLutXmp::Open(file_path, pass, cache_dir, true);
  
    if (!xmp) {
      std::cerr << xmp.error().message() << std::endl;
    }
  
    EXPECT_TRUE(xmp);
  
    std::cout << "               id: " << xmp->get_id() << std::endl;
    std::cout << " is photo enabled: " << xmp->is_photo_enabled() << std::endl;
    std::cout << " is video enabled: " << xmp->is_video_enabled() << std::endl;
    std::cout << "       color type: " << (int) xmp->get_color_type() << std::endl;
    std::cout << "        film type: " << (int) xmp->get_color_type() << std::endl;
  
    dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
    std::string ext = dehancer::TextureIO::extension_for(type);
    float compression = 1.0f;
    
    for (auto device: dehancer::DeviceCache::Instance().get_device_list()) {
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
      
      auto mlut = dehancer::CLutSquareInput(command_queue);
      
      mlut.load_from_image(xmp->get_cluts()[1]);
      
      std::string output_file =  "clut2d-"+platform+"-"+xmp->get_id()+"-";
      output_file.append(dehancer::device::get_name(device));
      output_file.append(ext);
  
      {
        std::ofstream ao_bench_os(output_file, std::ostream::binary | std::ostream::trunc);
        ao_bench_os << dehancer::TextureOutput(command_queue, mlut.get_texture(), {
                .type = type,
                .compression = compression
        });
      }
      
    }
  }
  
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw e;
  }
  
}
